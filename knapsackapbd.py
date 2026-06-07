# =============================================================
# PERBANDINGAN ALGORITMA BACKTRACKING, BRANCH AND BOUND, DAN
# SIMULATED ANNEALING DALAM OPTIMASI ALOKASI PROGRAM
# PEMBANGUNAN BERDASARKAN DAMPAK SOSIAL, BATASAN ANGGARAN
# SEKTORAL, DAN PEMERATAAN WILAYAH PADA SIMULASI APBD
# KOTA SEMARANG 2026

# =============================================================

# DESKRIPSI:
#  Masalah: Knapsack multidimensi dengan 2 constraint:
#    C1 — Total anggaran <= budget_cap (67% total dataset)
#    C2 — Tidak ada satu sektor > 35% total anggaran terpilih
#    HC — Program P01 (Pendidikan) dan P04 (Kesehatan) wajib masuk

#  Tiga algoritma dibandingkan:
#    1. Backtracking (BT)       — eksak, tanpa pruning intelijen
#    2. Branch and Bound (BnB)  — eksak, dengan upper bound pruning
#    3. Simulated Annealing (SA)— metaheuristik, probabilistik

#  Tiga ukuran dataset:
#    n=15 (kecil), n=25 (sedang), n=40 (penuh)


# =============================================================


import csv
import time
import math
import random
from dataclasses import dataclass, field
from typing import List, Dict, Tuple, Optional

# ─────────────────────────────────────────────
# 1. STRUKTUR DATA
# ─────────────────────────────────────────────

@dataclass
class Program:
    # Merepresentasikan satu program dalam APBD Kota Semarang 2026.
    id: str
    kode: str
    nama: str
    sektor: str
    kecamatan: str
    anggaran: float          # juta Rp
    penerima: float          # ribu jiwa
    urgensi: int             # 1–5
    skor: float              # skor dampak sosial 0–100
    hard_constraint: bool    # True = wajib masuk solusi

@dataclass
class HasilAlgoritma:
    # Menyimpan hasil satu run algoritma
    nama_algo: str
    skor_total: float
    anggaran_total: float
    items: List[Program]
    waktu_ms: float
    nodes_iter: int
    feasible: bool
    hc_terpenuhi: bool
    catatan: str = ""


# ─────────────────────────────────────────────
# 2. UTILITAS DATASET & CONSTRAINT
# ─────────────────────────────────────────────

MAX_SEKTOR_PCT = 0.35    # constraint sektor: maks 35% total anggaran terpilih

def muat_dataset(path: str, n: Optional[int] = None) -> List[Program]:
    
    # Muat dataset dari CSV.
    # Jika n ditentukan, ambil n program (hard constraint selalu masuk,
    # sisanya diambil dari yang skor tertinggi).
    progs: List[Program] = []
    with open(path, encoding="utf-8") as f:
        for row in csv.DictReader(f):
            progs.append(Program(
                id=row["id"],
                kode=row["kode"],
                nama=row["nama_program"],
                sektor=row["sektor"],
                kecamatan=row["kecamatan"],
                anggaran=float(row["anggaran_juta"]),
                penerima=float(row["penerima_ribu"]),
                urgensi=int(row["urgensi"]),
                skor=float(row["skor_dampak"]),
                hard_constraint=row["hard_constraint"].lower() == "true"
            ))
    # mengurutkan skor menurun agar BnB upper bound lebih efektif
    progs.sort(key=lambda p: p.skor, reverse=True)
    if n is not None:
        hc = [p for p in progs if p.hard_constraint]
        non_hc = [p for p in progs if not p.hard_constraint]
        sel = hc + non_hc[:n - len(hc)]
        sel.sort(key=lambda p: p.skor, reverse=True)
        return sel
    return progs

def cek_feasible(selected: List[Program], budget_cap: float) -> bool:
    """
    Periksa apakah solusi memenuhi semua constraint:
    - Total anggaran <= budget_cap
    - Tidak ada sektor > 35% total anggaran terpilih
    """
    if not selected:
        return True
    tot = sum(p.anggaran for p in selected)
    if tot > budget_cap:
        return False
    sektor_totals: Dict[str, float] = {}
    for p in selected:
        sektor_totals[p.sektor] = sektor_totals.get(p.sektor, 0.0) + p.anggaran
    for s, v in sektor_totals.items():
        if tot > 0 and v / tot > MAX_SEKTOR_PCT:
            return False
    return True

def total_skor(items: List[Program]) -> float:
    return sum(p.skor for p in items)

def total_anggaran(items: List[Program]) -> float:
    return sum(p.anggaran for p in items)


# ─────────────────────────────────────────────
# 3. ALGORITMA 1: BACKTRACKING
# ─────────────────────────────────────────────

def backtracking(programs: List[Program], budget_cap: float,
                 max_nodes: int = 5_000_000) -> HasilAlgoritma:
    """
    Backtracking murni: eksplorasi rekursif pohon keputusan biner.
    Setiap program memiliki 2 pilihan: dimasukkan (1) atau dilewati (0).

    Pruning dasar:
      - Jika total anggaran melebihi budget_cap, potong cabang kiri.
      - Hard constraint tidak boleh dilewati (tidak ada cabang kanan).

    Kompleksitas terburuk: O(2^n)
    Parameter max_nodes: batas eksplorasi agar tidak timeout pada n besar.
    """
    n = len(programs)
    hc_ids = {p.id for p in programs if p.hard_constraint}
    best: Dict = {"skor": 0.0, "items": [], "nodes": 0, "terpotong": False}

    def bt(idx: int, current: List[Program], curr_budget: float, curr_skor: float):
        if best["nodes"] >= max_nodes:
            best["terpotong"] = True
            return

        best["nodes"] += 1

        # Evaluasi solusi parsial yang sudah lengkap constraint-nya
        hc_sel = {p.id for p in current if p.hard_constraint}
        if hc_ids.issubset(hc_sel):
            if cek_feasible(current, budget_cap) and curr_skor > best["skor"]:
                best["skor"] = curr_skor
                best["items"] = current[:]

        if idx == n:
            return

        p = programs[idx]

        # Cabang KIRI — masukkan program ke-idx
        if curr_budget + p.anggaran <= budget_cap:
            current.append(p)
            bt(idx + 1, current, curr_budget + p.anggaran, curr_skor + p.skor)
            current.pop()

        # Cabang KANAN — lewati program ke-idx (tidak untuk hard constraint)
        if p.id not in hc_ids:
            bt(idx + 1, current, curr_budget, curr_skor)

    waktu0 = time.perf_counter()
    bt(0, [], 0.0, 0.0)
    waktu_ms = (time.perf_counter() - waktu0) * 1000

    items = best["items"]
    catatan = " [TERPOTONG — batas node tercapai]" if best["terpotong"] else " [OPTIMAL]"
    hc_ok = hc_ids.issubset({p.id for p in items})

    return HasilAlgoritma(
        nama_algo="Backtracking",
        skor_total=best["skor"],
        anggaran_total=total_anggaran(items),
        items=items,
        waktu_ms=waktu_ms,
        nodes_iter=best["nodes"],
        feasible=cek_feasible(items, budget_cap),
        hc_terpenuhi=hc_ok,
        catatan=catatan
    )


# ─────────────────────────────────────────────
# 4. ALGORITMA 2: BRANCH AND BOUND
# ─────────────────────────────────────────────

def branch_and_bound(programs: List[Program], budget_cap: float) -> HasilAlgoritma:
    """
    Branch and Bound dengan upper bound fractional knapsack.

    Upper bound dihitung sebagai:
      skor_sekarang + skor_fraksional_sisa_item
    (relaksasi LP — item terakhir boleh diambil sebagian)

    Pruning intelijen:
      - Jika upper_bound(node) <= best_skor, potong seluruh subtree.
      - Ini menghemat eksplorasi node secara dramatis vs Backtracking.

    Kompleksitas terburuk: O(2^n), praktis jauh lebih cepat.
    """
    n = len(programs)
    hc_ids = {p.id for p in programs if p.hard_constraint}
    best: Dict = {"skor": 0.0, "items": []}
    nodes = [0]

    def hitung_ub(idx: int, curr_budget: float, curr_skor: float) -> float:
        """Upper bound: fractional knapsack untuk item idx ke atas."""
        ub = curr_skor
        remaining = budget_cap - curr_budget
        for i in range(idx, n):
            p = programs[i]
            if remaining <= 0:
                break
            if p.anggaran <= remaining:
                ub += p.skor
                remaining -= p.anggaran
            else:
                # Ambil fraksional — ini adalah relaksasi LP
                ub += p.skor * (remaining / p.anggaran)
                break
        return ub

    def bnb(idx: int, current: List[Program], curr_budget: float, curr_skor: float):
        nodes[0] += 1

        # ★ PRUNING UTAMA: jika upper bound tidak bisa melampaui best, potong
        if hitung_ub(idx, curr_budget, curr_skor) <= best["skor"]:
            return

        # Evaluasi solusi
        hc_sel = {p.id for p in current if p.hard_constraint}
        if hc_ids.issubset(hc_sel):
            if cek_feasible(current, budget_cap) and curr_skor > best["skor"]:
                best["skor"] = curr_skor
                best["items"] = current[:]

        if idx == n:
            return

        p = programs[idx]

        # Cabang KIRI — ambil item
        if curr_budget + p.anggaran <= budget_cap:
            current.append(p)
            bnb(idx + 1, current, curr_budget + p.anggaran, curr_skor + p.skor)
            current.pop()

        # Cabang KANAN — lewati item (kecuali hard constraint)
        if p.id not in hc_ids:
            bnb(idx + 1, current, curr_budget, curr_skor)

    waktu0 = time.perf_counter()
    bnb(0, [], 0.0, 0.0)
    waktu_ms = (time.perf_counter() - waktu0) * 1000

    items = best["items"]
    hc_ok = hc_ids.issubset({p.id for p in items})

    return HasilAlgoritma(
        nama_algo="Branch and Bound",
        skor_total=best["skor"],
        anggaran_total=total_anggaran(items),
        items=items,
        waktu_ms=waktu_ms,
        nodes_iter=nodes[0],
        feasible=cek_feasible(items, budget_cap),
        hc_terpenuhi=hc_ok,
        catatan=" [OPTIMAL]"
    )


# ─────────────────────────────────────────────
# 5. ALGORITMA 3: SIMULATED ANNEALING
# ─────────────────────────────────────────────

def simulated_annealing(programs: List[Program], budget_cap: float,
                        T_start: float = 500.0, T_end: float = 0.01,
                        alpha: float = 0.995, max_iter: int = 80_000,
                        seed: int = 42) -> HasilAlgoritma:
    """
    Simulated Annealing — metaheuristik berbasis proses pendinginan logam.

    Representasi state: vektor biner bits[i] ∈ {0,1} untuk tiap program.
    Tetangga (neighbor): flip satu bit secara acak.
    Fungsi objektif: skor total solusi (penalti besar jika tidak feasible).
    Penerimaan solusi buruk: P(accept) = exp(delta/T) — semakin dingin T,
      semakin kecil kemungkinan menerima solusi yang lebih buruk.

    Keuntungan vs BT/BnB: tidak terjebak local optimum, sangat cepat
      untuk n besar, tapi tidak menjamin solusi optimal global.

    Parameter:
      T_start : suhu awal (tinggi = eksplorasi bebas)
      T_end   : suhu akhir (rendah = eksploitasi lokal)
      alpha   : laju pendinginan (T_baru = alpha * T_lama)
      max_iter: batas iterasi maksimum
    """
    random.seed(seed)
    n = len(programs)
    hc_indices = [i for i, p in enumerate(programs) if p.hard_constraint]
    non_hc_indices = [i for i in range(n) if i not in hc_indices]

    PENALTY = -1e9  # penalti untuk solusi yang melanggar constraint

    def evaluasi(bits: List[int]) -> float:
        selected = [programs[i] for i in range(n) if bits[i] == 1]
        if not selected:
            return 0.0
        tot = sum(p.anggaran for p in selected)
        if tot > budget_cap:
            return PENALTY
        sektor_tot: Dict[str, float] = {}
        for p in selected:
            sektor_tot[p.sektor] = sektor_tot.get(p.sektor, 0.0) + p.anggaran
        if any(tot > 0 and v / tot > MAX_SEKTOR_PCT for v in sektor_tot.values()):
            return PENALTY
        return sum(p.skor for p in selected)

    def tetangga(bits: List[int]) -> List[int]:
        """Flip satu bit random; hard constraint tidak boleh di-flip ke 0."""
        nb = bits[:]
        # Pilih secara acak antara: tambah item, hapus item, atau swap
        move = random.randint(0, 2)
        if move == 0 and non_hc_indices:
            # Flip satu bit non-HC
            idx = random.choice(non_hc_indices)
            nb[idx] = 1 - nb[idx]
        elif move == 1:
            # Hapus item non-HC yang sedang aktif
            aktif = [i for i in non_hc_indices if nb[i] == 1]
            if aktif:
                nb[random.choice(aktif)] = 0
        else:
            # Tambah item non-HC yang belum aktif
            tidak_aktif = [i for i in non_hc_indices if nb[i] == 0]
            if tidak_aktif:
                nb[random.choice(tidak_aktif)] = 1
        return nb

    # Inisialisasi: semua HC aktif, non-HC acak 50%
    current = [0] * n
    for i in hc_indices:
        current[i] = 1
    for i in non_hc_indices:
        current[i] = random.randint(0, 1)

    current_score = evaluasi(current)
    best_bits = current[:]
    best_score_sa = current_score
    T = T_start
    accepted = 0

    waktu0 = time.perf_counter()

    for iteration in range(max_iter):
        if T <= T_end:
            break
        nb = tetangga(current)
        nb_score = evaluasi(nb)
        delta = nb_score - current_score

        # Terima tetangga jika lebih baik, atau dengan probabilitas Boltzmann
        if delta > 0 or (T > 0 and random.random() < math.exp(delta / T)):
            current = nb
            current_score = nb_score
            accepted += 1
            if current_score > best_score_sa:
                best_score_sa = current_score
                best_bits = current[:]

        T *= alpha

    waktu_ms = (time.perf_counter() - waktu0) * 1000

    best_items = [programs[i] for i in range(n) if best_bits[i] == 1]
    hc_ids = {p.id for p in programs if p.hard_constraint}
    hc_ok = hc_ids.issubset({p.id for p in best_items})

    return HasilAlgoritma(
        nama_algo="Simulated Annealing",
        skor_total=best_score_sa,
        anggaran_total=total_anggaran(best_items),
        items=best_items,
        waktu_ms=waktu_ms,
        nodes_iter=max_iter,
        feasible=cek_feasible(best_items, budget_cap),
        hc_terpenuhi=hc_ok,
        catatan=f" [HEURISTIK | accepted={accepted:,}]"
    )


# ─────────────────────────────────────────────
# 6. RUNNER EKSPERIMEN
# ─────────────────────────────────────────────

def jalankan_eksperimen(programs: List[Program], label: str,
                        bt_max_nodes: int = 5_000_000) -> Dict[str, HasilAlgoritma]:
    """Jalankan ketiga algoritma pada dataset yang sama."""
    total_ds = total_anggaran(programs)
    budget_cap = total_ds * 0.67
    n = len(programs)

    SEP = "─" * 72
    print(f"\n╔{'═'*70}╗")
    print(f"║  EKSPERIMEN: {label:<56}║")
    print(f"║  n={n}, total dataset=Rp {total_ds:>12,.0f} juta{' '*19}║")
    print(f"║  budget_cap=Rp {budget_cap:>12,.0f} juta (67% total){' '*16}║")
    print(f"╚{'═'*70}╝")

    hasil: Dict[str, HasilAlgoritma] = {}

    # --- Backtracking ---
    print(f"\n  [1/3] Backtracking", end="", flush=True)
    if n > 22:
        print(f" (n={n}>22, dibatasi {bt_max_nodes:,} nodes) ...", end="", flush=True)
    else:
        print(" ...", end="", flush=True)
    r = backtracking(programs, budget_cap, max_nodes=bt_max_nodes)
    hasil["Backtracking"] = r
    print(f" {r.waktu_ms:.1f}ms | {r.nodes_iter:,} nodes{r.catatan}")

    # --- Branch and Bound ---
    print(f"  [2/3] Branch and Bound ...", end="", flush=True)
    r = branch_and_bound(programs, budget_cap)
    hasil["Branch and Bound"] = r
    print(f" {r.waktu_ms:.1f}ms | {r.nodes_iter:,} nodes{r.catatan}")

    # --- Simulated Annealing ---
    print(f"  [3/3] Simulated Annealing ...", end="", flush=True)
    r = simulated_annealing(programs, budget_cap)
    hasil["Simulated Annealing"] = r
    print(f" {r.waktu_ms:.1f}ms | {r.nodes_iter:,} iterasi{r.catatan}")

    # ── Tabel Hasil ──
    print(f"\n  {'Algoritma':<22} {'Skor':>8} {'Anggaran(jt)':>15} {'Items':>6} "
          f"{'Waktu(ms)':>11} {'Nodes/Iter':>12} {'Feasible':>9} {'HC':>5}")
    print(f"  {SEP}")
    for algo, r in hasil.items():
        fs = "✓" if r.feasible else "✗"
        hc = "✓" if r.hc_terpenuhi else "✗"
        print(f"  {algo:<22} {r.skor_total:>8.1f} {r.anggaran_total:>15,.0f} "
              f"{len(r.items):>6} {r.waktu_ms:>11.2f} {r.nodes_iter:>12,d} "
              f"{fs:>9} {hc:>5}")

    # ── Analisis Komparatif ──
    bt_r  = hasil["Backtracking"]
    bnb_r = hasil["Branch and Bound"]
    sa_r  = hasil["Simulated Annealing"]
    opt   = max(bt_r.skor_total, bnb_r.skor_total)

    print(f"\n  ┌─ Analisis ──────────────────────────────────────────────────────")
    if not bt_r.catatan.__contains__("TERPOTONG") and bt_r.waktu_ms > 0:
        speedup = bt_r.waktu_ms / max(bnb_r.waktu_ms, 0.001)
        reduksi = (1 - bnb_r.nodes_iter / max(bt_r.nodes_iter, 1)) * 100
        print(f"  │  BnB vs BT  : Speedup {speedup:.1f}x | Reduksi node {reduksi:.1f}%")
    else:
        print(f"  │  BnB vs BT  : BT terpotong di {bt_r.nodes_iter:,} nodes "
              f"(n={n} terlalu besar untuk BT eksak)")
    if opt > 0:
        gap = (opt - sa_r.skor_total) / opt * 100
        print(f"  │  Gap SA     : {gap:.2f}% dari solusi optimal (BnB)")
    print(f"  └─────────────────────────────────────────────────────────────────")

    # ── Detail Solusi BnB ──
    best_items = bnb_r.items
    print(f"\n  Solusi terpilih (Branch and Bound) — {len(best_items)} program:")
    for p in sorted(best_items, key=lambda x: x.skor, reverse=True):
        hc_m = "★" if p.hard_constraint else " "
        print(f"    {hc_m} {p.id} | Skor {p.skor:>5.1f} | "
              f"Rp {p.anggaran:>9,.0f} jt | {p.nama[:56]}")

    return hasil


# ─────────────────────────────────────────────
# 7. MAIN
# ─────────────────────────────────────────────

if __name__ == "__main__":
    DATA = "dataset_apbd_semarang_2026.csv"

    print("╔══════════════════════════════════════════════════════════════════════╗")
    print("║  OPTIMASI ALOKASI PROGRAM APBD KOTA SEMARANG 2026                   ║")
    print("║  Algoritma: Backtracking | Branch and Bound | Simulated Annealing   ║")
    print("║  Dataset  : Perda Kota Semarang No. 13 Tahun 2025 (Hybrid)          ║")
    print("╚══════════════════════════════════════════════════════════════════════╝")

    semua_hasil: Dict[str, Dict] = {}

    # Eksperimen 1 — Dataset Kecil (n=15)
    prog_15 = muat_dataset(DATA, n=15)
    semua_hasil["n=15"] = jalankan_eksperimen(prog_15, "Dataset Kecil (n=15)")

    # Eksperimen 2 — Dataset Sedang (n=25)
    prog_25 = muat_dataset(DATA, n=25)
    semua_hasil["n=25"] = jalankan_eksperimen(prog_25, "Dataset Sedang (n=25)")

    # Eksperimen 3 — Dataset Penuh (n=40)
    prog_40 = muat_dataset(DATA, n=40)
    semua_hasil["n=40"] = jalankan_eksperimen(prog_40, "Dataset Penuh (n=40)")

    # ── TABEL RINGKASAN AKHIR ──
    print(f"\n\n{'═'*72}")
    print("  TABEL RINGKASAN KOMPARATIF — LINTAS UKURAN DATASET")
    print(f"{'═'*72}")
    print(f"  {'Dataset':<8} {'Algoritma':<22} {'Skor':>8} {'Waktu(ms)':>11} "
          f"{'Nodes/Iter':>12} {'Feasible':>9}")
    print(f"  {'─'*68}")
    for ds_lbl, hasil in semua_hasil.items():
        for algo, r in hasil.items():
            fs = "✓" if r.feasible else "✗"
            print(f"  {ds_lbl:<8} {algo:<22} {r.skor_total:>8.1f} "
                  f"{r.waktu_ms:>11.2f} {r.nodes_iter:>12,d} {fs:>9}")
        print()

    print("  ★ = Hard Constraint (wajib masuk: P01 Pendidikan, P04 Kesehatan)")