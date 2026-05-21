"""
CUDA Naive GEMM Educational Animation
Style: 3Blue1Brown / Manim Community Edition

Render with:
    manim -pql cuda_gemm_animation.py CUDAGEMMAnimation
    # or for high quality:
    manim -pqh cuda_gemm_animation.py CUDAGEMMAnimation

Each Scene class is a standalone chapter. The combined CUDAGEMMAnimation
class plays them all in sequence.
"""

from manim import *
import numpy as np

# ─────────────────────────────────────────────────────────────────────────────
# GLOBAL STYLE CONSTANTS  (3b1b-inspired palette)
# ─────────────────────────────────────────────────────────────────────────────
BG_COLOR      = "#0f0f1a"
ACCENT_BLUE   = "#4FC3F7"
ACCENT_GOLD   = "#FFD54F"
ACCENT_GREEN  = "#A5D6A7"
ACCENT_RED    = "#EF9A9A"
ACCENT_PURPLE = "#CE93D8"
ACCENT_TEAL   = "#80CBC4"
DIM_WHITE     = "#CCCCCC"
CODE_BG       = "#1a1a2e"

config.background_color = BG_COLOR


# ─────────────────────────────────────────────────────────────────────────────
# HELPER UTILITIES
# ─────────────────────────────────────────────────────────────────────────────

def make_matrix_mob(rows, cols, cell_size=0.55, fill=None,
                    border_color=ACCENT_BLUE, border_width=1.5,
                    label_fn=None):
    """
    Return a VGroup of Rectangle cells arranged as a grid.
    label_fn(r, c) -> string label for cell (r,c), or None for no label.
    fill can be a 2-D list of colors.
    """
    grid = VGroup()
    cells = []
    for r in range(rows):
        row_cells = []
        for c in range(cols):
            color = fill[r][c] if fill else "#1a1a2e"
            cell = Rectangle(
                width=cell_size, height=cell_size,
                fill_color=color, fill_opacity=0.85,
                stroke_color=border_color, stroke_width=border_width,
            )
            cell.move_to(RIGHT * c * cell_size + DOWN * r * cell_size)
            if label_fn:
                lbl = label_fn(r, c)
                if lbl:
                    txt = Text(lbl, font_size=12, color=WHITE).move_to(cell)
                    grid.add(cell, txt)
            else:
                grid.add(cell)
            row_cells.append(cell)
        cells.append(row_cells)
    grid.center()
    return grid, cells


def chapter_title(scene, title_text, subtitle_text=""):
    """Flash a chapter title card."""
    t = Text(title_text, font_size=52, color=ACCENT_GOLD, weight=BOLD)
    s = Text(subtitle_text, font_size=28, color=DIM_WHITE)
    VGroup(t, s).arrange(DOWN, buff=0.4).center()
    scene.play(FadeIn(t, scale=0.85), run_time=0.8)
    if subtitle_text:
        scene.play(FadeIn(s), run_time=0.5)
    scene.wait(1.5)
    scene.play(FadeOut(t), FadeOut(s), run_time=0.5)


def narration(scene, text, duration=2.5, font_size=26, color=DIM_WHITE,
              position=DOWN * 3.3):
    """Show a narration line at the bottom, hold, then fade it."""
    mob = Text(text, font_size=font_size, color=color,
               line_spacing=1.3).move_to(position)
    scene.play(FadeIn(mob, shift=UP * 0.15), run_time=0.45)
    scene.wait(duration)
    scene.play(FadeOut(mob), run_time=0.3)
    return mob


def glow_rect(mob, color=ACCENT_GOLD, opacity=0.35):
    """Return a glowing rectangle overlay on top of a mobject."""
    r = SurroundingRectangle(mob, color=color, buff=0.05, stroke_width=3)
    r.set_fill(color=color, opacity=opacity)
    return r


# ─────────────────────────────────────────────────────────────────────────────
# SCENE 1 — INTRODUCTION
# ─────────────────────────────────────────────────────────────────────────────
class S01_Introduction(Scene):
    def construct(self):
        self.camera.background_color = BG_COLOR

        # ── Opening question ──────────────────────────────────────────────
        q = Text("How can thousands of GPU threads\ncooperate to multiply matrices?",
                 font_size=42, color=ACCENT_GOLD, line_spacing=1.4).center()
        self.play(Write(q), run_time=2.5)
        self.wait(2)
        self.play(FadeOut(q))

        # ── Show large matrix multiplication C = A × B ────────────────────
        M, K, N = 6, 5, 6
        cell = 0.42

        def make_mini(rows, cols, color, label):
            g = VGroup()
            for r in range(rows):
                for c in range(cols):
                    rect = Rectangle(width=cell, height=cell,
                                     fill_color=color, fill_opacity=0.25,
                                     stroke_color=color, stroke_width=1.2)
                    rect.move_to(RIGHT * c * cell + DOWN * r * cell)
                    g.add(rect)
            g.center()
            lbl = Text(label, font_size=26, color=color).next_to(g, UP, buff=0.18)
            return VGroup(g, lbl)

        A_mob = make_mini(M, K, ACCENT_BLUE,  "A  (M×K)")
        B_mob = make_mini(K, N, ACCENT_GREEN, "B  (K×N)")
        C_mob = make_mini(M, N, ACCENT_GOLD,  "C  (M×N)")

        eq = Text("=", font_size=48, color=WHITE)
        times = Text("×", font_size=44, color=WHITE)

        full = VGroup(A_mob, times, B_mob, eq, C_mob).arrange(RIGHT, buff=0.45)
        full.scale_to_fit_width(12).center()

        self.play(FadeIn(A_mob, shift=LEFT * 0.3), run_time=0.7)
        self.play(FadeIn(times), FadeIn(B_mob, shift=LEFT * 0.3), run_time=0.7)
        self.play(Write(eq), FadeIn(C_mob, shift=LEFT * 0.3), run_time=0.7)
        self.wait(1)

        narration(self,
            "Matrix multiplication: multiply A and B to get C.",
            duration=2)

        # ── Zoom into one output element ───────────────────────────────────
        # Highlight C[1][2] cell
        c_grid = C_mob[0]
        target_cell_idx = 1 * N + 2   # row=1, col=2
        focus_cell = c_grid[target_cell_idx]

        glow = glow_rect(focus_cell, ACCENT_GOLD, 0.7)
        self.play(Create(glow), run_time=0.6)
        narration(self,
            "Each cell in C is one output element — let's zoom in.",
            duration=2)

        self.play(FadeOut(glow))

        # Key insight
        insight = Text(
            "What if ONE thread computes ONE output element?",
            font_size=36, color=ACCENT_TEAL).center()
        self.play(FadeOut(full))
        self.play(Write(insight), run_time=1.5)
        self.wait(2.5)
        self.play(FadeOut(insight))


# ─────────────────────────────────────────────────────────────────────────────
# SCENE 2 — MATRIX MULTIPLICATION INTUITION
# ─────────────────────────────────────────────────────────────────────────────
class S02_MatMulIntuition(Scene):
    def construct(self):
        self.camera.background_color = BG_COLOR
        chapter_title(self, "Matrix Multiplication", "Row × Column = One Output Cell")

        M, K, N = 4, 4, 4
        cell = 0.55

        # Build A and B grids
        a_cells, b_cells, c_cells = [], [], []

        def make_grid(rows, cols, default_color, stroke_col):
            vg = VGroup()
            cell_list = []
            for r in range(rows):
                row = []
                for c in range(cols):
                    rect = Rectangle(width=cell, height=cell,
                                     fill_color=default_color, fill_opacity=0.2,
                                     stroke_color=stroke_col, stroke_width=1.5)
                    rect.move_to(RIGHT * c * cell + DOWN * r * cell)
                    vg.add(rect)
                    row.append(rect)
                cell_list.append(row)
            vg.center()
            return vg, cell_list

        A_vg, a_cells = make_grid(M, K, ACCENT_BLUE, ACCENT_BLUE)
        B_vg, b_cells = make_grid(K, N, ACCENT_GREEN, ACCENT_GREEN)
        C_vg, c_cells = make_grid(M, N, ACCENT_GOLD, ACCENT_GOLD)

        lA = Text("A", font_size=30, color=ACCENT_BLUE).next_to(A_vg, UP, buff=0.15)
        lB = Text("B", font_size=30, color=ACCENT_GREEN).next_to(B_vg, UP, buff=0.15)
        lC = Text("C", font_size=30, color=ACCENT_GOLD).next_to(C_vg, UP, buff=0.15)

        grp = VGroup(
            VGroup(A_vg, lA),
            VGroup(B_vg, lB),
            VGroup(C_vg, lC)
        ).arrange(RIGHT, buff=1.1).shift(UP * 0.3)

        self.play(FadeIn(grp))
        self.wait(0.5)

        # ── Highlight row 1 of A and col 1 of B ───────────────────────────
        target_row = 1
        target_col = 1

        narration(self,
            "To compute C[1][1], we take row 1 of A ...",
            duration=2.2)

        row_highlights = []
        for c in range(K):
            h = a_cells[target_row][c].copy()
            h.set_fill(ACCENT_BLUE, opacity=0.85)
            h.set_stroke(ACCENT_BLUE, width=3)
            row_highlights.append(h)
        self.play(*[FadeIn(h) for h in row_highlights], run_time=0.6)
        self.wait(0.8)

        narration(self,
            "... and column 1 of B.",
            duration=2)

        col_highlights = []
        for r in range(K):
            h = b_cells[r][target_col].copy()
            h.set_fill(ACCENT_GREEN, opacity=0.85)
            h.set_stroke(ACCENT_GREEN, width=3)
            col_highlights.append(h)
        self.play(*[FadeIn(h) for h in col_highlights], run_time=0.6)
        self.wait(0.8)

        # ── Animate the dot product k=0..3 ────────────────────────────────
        narration(self,
            "Multiply element-by-element and sum — that's the dot product.",
            duration=2.5)

        formula_parts = []
        for k in range(K):
            a_h = row_highlights[k]
            b_h = col_highlights[k]
            # Flash pair
            self.play(
                a_h.animate.set_fill(ACCENT_GOLD, opacity=1.0),
                b_h.animate.set_fill(ACCENT_GOLD, opacity=1.0),
                run_time=0.25
            )
            self.play(
                a_h.animate.set_fill(ACCENT_BLUE, opacity=0.85),
                b_h.animate.set_fill(ACCENT_GREEN, opacity=0.85),
                run_time=0.2
            )

        # Highlight the output cell
        out_h = c_cells[target_row][target_col].copy()
        out_h.set_fill(ACCENT_GOLD, opacity=1.0)
        out_h.set_stroke(ACCENT_GOLD, width=3)
        self.play(FadeIn(out_h), run_time=0.5)

        # Formula
        formula = MathTex(
            r"C[1][1] = \sum_{k=0}^{K-1} A[1][k] \cdot B[k][1]",
            font_size=34, color=WHITE
        ).next_to(grp, DOWN, buff=0.55)
        self.play(Write(formula))
        self.wait(2.5)

        self.play(FadeOut(grp), FadeOut(formula),
                  *[FadeOut(h) for h in row_highlights + col_highlights],
                  FadeOut(out_h))


# ─────────────────────────────────────────────────────────────────────────────
# SCENE 3 — APARTMENT ANALOGY (GPU THREADS / BLOCKS / GRIDS)
# ─────────────────────────────────────────────────────────────────────────────
class S03_ApartmentAnalogy(Scene):
    def construct(self):
        self.camera.background_color = BG_COLOR
        chapter_title(self, "GPU Threads", "The Apartment Complex Analogy")

        # ── City → Grid, Buildings → Blocks, Apartments → Threads ─────────
        city_label  = Text("🌆  City   =  Grid",    font_size=34, color=ACCENT_GOLD)
        build_label = Text("🏢  Building  =  Block", font_size=34, color=ACCENT_BLUE)
        apt_label   = Text("🚪  Apartment  =  Thread", font_size=34, color=ACCENT_GREEN)

        labels = VGroup(city_label, build_label, apt_label).arrange(DOWN, buff=0.5)
        labels.shift(LEFT * 3.5)

        self.play(FadeIn(city_label, shift=RIGHT * 0.3), run_time=0.7)
        self.play(FadeIn(build_label, shift=RIGHT * 0.3), run_time=0.7)
        self.play(FadeIn(apt_label, shift=RIGHT * 0.3), run_time=0.7)
        self.wait(1)

        # ── Draw two buildings side-by-side, each with 3 apartments ───────
        # Building 0
        def draw_building(block_x, color, label):
            apts = VGroup()
            apt_cells = []
            for i in range(3):
                apt = Rectangle(width=0.9, height=0.65,
                                fill_color=color, fill_opacity=0.25,
                                stroke_color=color, stroke_width=2)
                apt.move_to(RIGHT * i * 0.95)
                apt_num = Text(str(i), font_size=18, color=color).move_to(apt)
                apts.add(apt, apt_num)
                apt_cells.append(apt)
            bld = SurroundingRectangle(apts, color=color, buff=0.15,
                                       stroke_width=2.5)
            lbl = Text(label, font_size=20, color=color).next_to(bld, UP, buff=0.1)
            grp = VGroup(apts, bld, lbl).shift(RIGHT * block_x)
            return grp, apt_cells

        b0, b0_cells = draw_building(-1.0, ACCENT_BLUE, "Block 0")
        b1, b1_cells = draw_building( 3.0, ACCENT_PURPLE, "Block 1")

        self.play(FadeOut(labels))
        self.play(FadeIn(b0), run_time=0.7)
        self.play(FadeIn(b1), run_time=0.7)
        self.wait(0.5)

        narration(self,
            "Every building (block) has apartments numbered 0, 1, 2 ...",
            duration=2.5)

        # Show repeated apartment numbering
        repeat_note = Text("threadIdx.x repeats inside every block!",
                           font_size=26, color=ACCENT_GOLD)
        repeat_note.next_to(VGroup(b0, b1), DOWN, buff=0.55)
        self.play(FadeIn(repeat_note))
        self.wait(2)
        self.play(FadeOut(repeat_note))

        # ── Deriving global address ────────────────────────────────────────
        narration(self,
            "To find someone globally, we need: building offset + apartment number.",
            duration=2.8)

        derive_title = Text("Global address derivation", font_size=30,
                            color=ACCENT_TEAL).to_edge(UP).shift(DOWN * 0.3)
        self.play(FadeIn(derive_title))

        step1 = MathTex(r"\text{block offset} = \text{blockIdx.x} \times \text{blockDim.x}",
                        font_size=32, color=ACCENT_BLUE)
        step2 = MathTex(r"\text{global\_x} = \text{block offset} + \text{threadIdx.x}",
                        font_size=32, color=ACCENT_GREEN)
        step3 = MathTex(r"\Rightarrow \quad \text{global\_x} = \text{blockIdx.x} \cdot \text{blockDim.x} + \text{threadIdx.x}",
                        font_size=30, color=ACCENT_GOLD)

        VGroup(step1, step2, step3).arrange(DOWN, buff=0.45).shift(DOWN * 0.8)

        self.play(FadeOut(b0), FadeOut(b1))

        # Concrete example
        example = Text("Example: blockIdx.x=1, blockDim.x=3, threadIdx.x=2",
                       font_size=24, color=DIM_WHITE).next_to(step3, DOWN, buff=0.5)
        result = MathTex(r"\text{global\_x} = 1 \times 3 + 2 = 5",
                         font_size=30, color=ACCENT_GOLD).next_to(example, DOWN, buff=0.3)

        self.play(Write(step1), run_time=1.2)
        self.wait(0.8)
        self.play(Write(step2), run_time=1.2)
        self.wait(0.8)
        self.play(Write(step3), run_time=1.5)
        self.wait(1)
        self.play(FadeIn(example), Write(result), run_time=1.2)
        self.wait(2.5)

        self.play(FadeOut(VGroup(derive_title, step1, step2, step3, example, result)))


# ─────────────────────────────────────────────────────────────────────────────
# SCENE 4 — BLOCKS AND GRIDS  (2-D case)
# ─────────────────────────────────────────────────────────────────────────────
class S04_BlocksAndGrids(Scene):
    def construct(self):
        self.camera.background_color = BG_COLOR
        chapter_title(self, "Blocks & Grids", "Tiling the Output Matrix")

        # dim3 block(3,3), dim3 grid(3,3) → 9×9 output
        GRID_BLOCKS  = 3   # blocks per dimension
        THREADS_PER  = 3   # threads per block per dimension
        cell_sz = 0.38
        block_gap = 0.10

        # Color palette for 9 blocks
        block_colors = [
            ACCENT_BLUE, ACCENT_GREEN, ACCENT_PURPLE,
            ACCENT_GOLD, ACCENT_TEAL, ACCENT_RED,
            "#82B1FF", "#FFCCBC", "#B9F6CA"
        ]

        title = Text("Grid (3×3 blocks)  ·  Each block has 3×3 threads",
                     font_size=28, color=DIM_WHITE).to_edge(UP).shift(DOWN * 0.3)
        self.play(FadeIn(title))

        all_cells = VGroup()
        block_groups = []

        for bx in range(GRID_BLOCKS):
            for by in range(GRID_BLOCKS):
                block_idx = bx * GRID_BLOCKS + by
                color = block_colors[block_idx]
                block_cells = VGroup()
                for tx in range(THREADS_PER):
                    for ty in range(THREADS_PER):
                        row = bx * THREADS_PER + tx
                        col = by * THREADS_PER + ty
                        x_pos = col * (cell_sz + 0.01) + by * block_gap
                        y_pos = -row * (cell_sz + 0.01) - bx * block_gap
                        rect = Rectangle(
                            width=cell_sz, height=cell_sz,
                            fill_color=color, fill_opacity=0.30,
                            stroke_color=color, stroke_width=1.0
                        ).move_to(RIGHT * x_pos + UP * y_pos)
                        block_cells.add(rect)
                        all_cells.add(rect)
                block_groups.append(block_cells)

        all_cells.center().shift(RIGHT * 0.5)

        # Animate blocks appearing one-by-one
        self.play(FadeIn(all_cells[0:9]), run_time=0.4)
        for i, bg in enumerate(block_groups):
            self.play(FadeIn(bg), run_time=0.18)
        self.wait(0.5)

        # Draw thick borders around each block
        block_borders = VGroup()
        for i, bg in enumerate(block_groups):
            border = SurroundingRectangle(bg,
                color=block_colors[i], buff=0.03, stroke_width=2.5)
            block_borders.add(border)
        self.play(Create(block_borders), run_time=1.2)

        narration(self,
            "Each colored region is one block. Every tiny square is one thread.",
            duration=3)

        # Labels
        block_lbl = Text("Block", font_size=22, color=ACCENT_GOLD)
        arr1 = Arrow(ORIGIN, RIGHT * 1.2, color=ACCENT_GOLD, buff=0.05)
        block_lbl.next_to(block_borders[4], LEFT, buff=0.3)
        self.play(FadeIn(block_lbl), run_time=0.5)

        thread_lbl = Text("Thread", font_size=18, color=DIM_WHITE)
        thread_lbl.next_to(all_cells, RIGHT, buff=0.4)
        self.play(FadeIn(thread_lbl), run_time=0.5)
        self.wait(2)

        # Code
        code_txt = Code(
            code_string="""dim3 blockDim(3, 3);   // 3×3 threads per block
dim3 gridDim(3, 3);    // 3×3 blocks in grid
// Total threads: 9×9 = 81""",
            language="cpp",
            background="rectangle",
            background_config={"color": ACCENT_BLUE, "stroke_width": 2},
            paragraph_config={"font_size": 22},
            add_line_numbers=False,
        ).to_edge(DOWN).shift(UP * 0.25)

        self.play(FadeIn(code_txt))
        self.wait(2.5)
        self.play(FadeOut(VGroup(all_cells, block_borders, title,
                                  block_lbl, thread_lbl, code_txt)))


# ─────────────────────────────────────────────────────────────────────────────
# SCENE 5 — THREAD MAPPING  (concrete 9×9 example)
# ─────────────────────────────────────────────────────────────────────────────
class S05_ThreadMapping(Scene):
    def construct(self):
        self.camera.background_color = BG_COLOR
        chapter_title(self, "Thread Mapping", "Who computes which cell?")

        SIZE = 9
        BLOCK = 3
        cell_sz = 0.50

        # Draw the 9×9 output grid
        cells = {}
        grid_vg = VGroup()
        for r in range(SIZE):
            for c in range(SIZE):
                rect = Rectangle(
                    width=cell_sz, height=cell_sz,
                    fill_color="#111122", fill_opacity=0.95,
                    stroke_color=ACCENT_BLUE, stroke_width=0.8
                ).move_to(RIGHT * c * cell_sz + DOWN * r * cell_sz)
                cells[(r, c)] = rect
                grid_vg.add(rect)
        grid_vg.center()

        self.play(FadeIn(grid_vg), run_time=0.8)

        narration(self,
            "9×9 output matrix. Block size 3×3. Grid size 3×3.",
            duration=2.5)

        # Highlight block (1,2): rows 3-5, cols 6-8
        BX, BY = 1, 2
        block_cells = VGroup(*[cells[(BX*BLOCK+tx, BY*BLOCK+ty)]
                                for tx in range(BLOCK) for ty in range(BLOCK)])
        block_border = SurroundingRectangle(block_cells, color=ACCENT_PURPLE,
                                            buff=0.03, stroke_width=3)
        block_label = Text("Block (1,2)", font_size=22, color=ACCENT_PURPLE)
        block_label.next_to(block_border, UP, buff=0.15)

        self.play(
            *[c.animate.set_fill(ACCENT_PURPLE, opacity=0.35)
              for c in block_cells],
            Create(block_border), FadeIn(block_label),
            run_time=0.8
        )
        self.wait(1)

        # Highlight thread (1,2) inside block (1,2) → global (4, 8)
        TX, TY = 1, 2
        global_r = BX * BLOCK + TX   # = 1*3 + 1 = 4
        global_c = BY * BLOCK + TY   # = 2*3 + 2 = 8

        target = cells[(global_r, global_c)]
        glow = glow_rect(target, ACCENT_GOLD, 0.9)

        self.play(Create(glow))

        # Derivation annotation
        deriv = VGroup(
            Text("Thread (tx=1, ty=2) inside Block (bx=1, by=2)", font_size=22,
                 color=DIM_WHITE),
            MathTex(r"\text{row} = b_x \cdot \text{blockDim.x} + t_x = 1 \cdot 3 + 1 = 4",
                    font_size=26, color=ACCENT_BLUE),
            MathTex(r"\text{col} = b_y \cdot \text{blockDim.y} + t_y = 2 \cdot 3 + 2 = 8",
                    font_size=26, color=ACCENT_GREEN),
        ).arrange(DOWN, buff=0.3).to_edge(DOWN).shift(UP * 0.3)

        self.play(FadeIn(deriv[0]))
        self.wait(0.5)
        self.play(Write(deriv[1]))
        self.wait(0.5)
        self.play(Write(deriv[2]))
        self.wait(2.5)

        # CUDA code
        cuda_line = Code(
            code_string="""const uint x = blockIdx.x * blockDim.x + threadIdx.x;  // row
const uint y = blockIdx.y * blockDim.y + threadIdx.y;  // col""",
            language="cpp",
            background="rectangle",
            background_config={"color": ACCENT_GOLD, "stroke_width": 2},
            paragraph_config={"font_size": 22},
            add_line_numbers=False,
        ).to_edge(UP).shift(DOWN * 0.3)

        self.play(FadeIn(cuda_line))
        self.wait(2.5)
        self.play(FadeOut(VGroup(grid_vg, block_border, block_label, glow,
                                  deriv, cuda_line)))


# ─────────────────────────────────────────────────────────────────────────────
# SCENE 6 — CEILING DIVISION & BOUNDS CHECK
# ─────────────────────────────────────────────────────────────────────────────
class S06_CeilingDivision(Scene):
    def construct(self):
        self.camera.background_color = BG_COLOR
        chapter_title(self, "Ceiling Division", "What if the matrix doesn't fit perfectly?")

        # Visual: 9 columns, block width 4 → ceil(9/4) = 3 blocks
        N = 9
        BLOCK_W = 4
        import math
        num_blocks = math.ceil(N / BLOCK_W)   # = 3

        cell_h = 0.65
        cell_w = 0.62
        y_base = 0.8

        # Draw the 9 matrix columns
        mat_cells = VGroup()
        for c in range(N):
            rect = Rectangle(width=cell_w, height=cell_h,
                             fill_color=ACCENT_BLUE, fill_opacity=0.3,
                             stroke_color=ACCENT_BLUE, stroke_width=1.5)
            rect.move_to(RIGHT * c * (cell_w + 0.05) + UP * y_base)
            mat_cells.add(rect)
        mat_cells.center()

        mat_lbl = Text(f"Matrix: {N} columns", font_size=26, color=ACCENT_BLUE)
        mat_lbl.next_to(mat_cells, UP, buff=0.2)

        self.play(FadeIn(mat_cells), FadeIn(mat_lbl), run_time=0.8)
        self.wait(0.5)

        narration(self,
            "We have 9 columns, but our block width is 4.",
            duration=2.5)

        # Draw 2 blocks (insufficient)
        block_colors_local = [ACCENT_GREEN, ACCENT_PURPLE, ACCENT_GOLD]
        block_mobs = []
        labels_b = []
        for b in range(num_blocks):
            start = b * BLOCK_W
            end   = min(start + BLOCK_W, 100)   # extend beyond matrix for ghost
            w = BLOCK_W * (cell_w + 0.05) - 0.05
            x_center = mat_cells[0].get_center()[0] + (start + BLOCK_W/2 - 0.5) * (cell_w + 0.05)
            y_pos = y_base - cell_h - 0.35

            ghost = Rectangle(
                width=w, height=cell_h * 0.6,
                fill_color=block_colors_local[b], fill_opacity=0.20,
                stroke_color=block_colors_local[b], stroke_width=2.0
            ).move_to(RIGHT * x_center + UP * y_pos)

            lbl = Text(f"Block {b}", font_size=18, color=block_colors_local[b])
            lbl.next_to(ghost, DOWN, buff=0.08)

            block_mobs.append(ghost)
            labels_b.append(lbl)

        self.play(*[FadeIn(b) for b in block_mobs[:2]],
                  *[FadeIn(l) for l in labels_b[:2]], run_time=0.7)

        narration(self,
            "2 blocks cover only 8 columns — we miss the last one!",
            duration=2.5)

        # Cross mark
        cross = Text("✗ Only 8 covered", font_size=24, color=ACCENT_RED)
        cross.next_to(mat_cells, DOWN, buff=1.4)
        self.play(FadeIn(cross))
        self.wait(1.5)
        self.play(FadeOut(cross))

        # Add block 2
        self.play(FadeIn(block_mobs[2]), FadeIn(labels_b[2]), run_time=0.6)

        # Show the "ghost" cells extending beyond matrix
        ghost_cells = VGroup()
        for c in range(N, num_blocks * BLOCK_W):
            rect = Rectangle(width=cell_w, height=cell_h,
                             fill_color=ACCENT_RED, fill_opacity=0.25,
                             stroke_color=ACCENT_RED, stroke_width=1.5, stroke_opacity=0.5)
            offset = mat_cells[-1].get_right()[0] + (c - N + 0.5) * (cell_w + 0.05)
            rect.move_to(RIGHT * offset + UP * y_base)
            ghost_cells.add(rect)

        if len(ghost_cells) > 0:
            ghost_lbl = Text("Out-of-bounds threads", font_size=20, color=ACCENT_RED)
            ghost_lbl.next_to(ghost_cells, UP, buff=0.15)
            self.play(FadeIn(ghost_cells), FadeIn(ghost_lbl))
            self.wait(1)

        narration(self,
            "Block 3 partially extends outside. We need a bounds check!",
            duration=2.5)

        # The formula
        formula_vg = VGroup(
            MathTex(r"\text{num\_blocks} = \left\lceil \frac{N}{\text{blockSize}} \right\rceil",
                    font_size=36, color=ACCENT_GOLD),
            MathTex(r"= \frac{N + \text{blockSize} - 1}{\text{blockSize}}",
                    font_size=32, color=ACCENT_GREEN),
        ).arrange(DOWN, buff=0.3).to_edge(DOWN).shift(UP * 0.6)

        self.play(Write(formula_vg[0]))
        self.wait(0.8)
        self.play(Write(formula_vg[1]))
        self.wait(1.5)

        bounds_check = Code(
            code_string="if (x < M && y < N) { /* valid thread */ }",
            language="cpp",
            background="rectangle",
            background_config={"color": ACCENT_RED, "stroke_width": 2},
            paragraph_config={"font_size": 24},
            add_line_numbers=False,
        ).next_to(formula_vg, DOWN, buff=0.35)

        bounds_note = Text('"Ignore apartments outside the city limits"',
                           font_size=22, color=ACCENT_RED)
        bounds_note.next_to(bounds_check, DOWN, buff=0.2)

        self.play(FadeIn(bounds_check))
        self.play(FadeIn(bounds_note))
        self.wait(2.5)

        self.play(FadeOut(VGroup(mat_cells, mat_lbl, formula_vg,
                                  bounds_check, bounds_note,
                                  *block_mobs, *labels_b)))
        if len(ghost_cells) > 0:
            self.play(FadeOut(ghost_cells), FadeOut(ghost_lbl))


# ─────────────────────────────────────────────────────────────────────────────
# SCENE 7 — MEMORY FLATTENING
# ─────────────────────────────────────────────────────────────────────────────
class S07_MemoryFlattening(Scene):
    def construct(self):
        self.camera.background_color = BG_COLOR
        chapter_title(self, "Memory Flattening", "2-D matrix → 1-D memory")

        ROWS, COLS = 3, 4
        cell_sz = 0.70

        # ── 2-D grid ──────────────────────────────────────────────────────
        grid2d = VGroup()
        grid_cells = {}
        for r in range(ROWS):
            for c in range(COLS):
                rect = Rectangle(
                    width=cell_sz, height=cell_sz,
                    fill_color=ACCENT_BLUE, fill_opacity=0.25,
                    stroke_color=ACCENT_BLUE, stroke_width=1.5
                ).move_to(RIGHT * c * (cell_sz + 0.05) + DOWN * r * (cell_sz + 0.05))
                lbl = Text(f"[{r}][{c}]", font_size=14, color=ACCENT_BLUE).move_to(rect)
                grid2d.add(rect, lbl)
                grid_cells[(r, c)] = rect
        grid2d.center().shift(UP * 1.2)

        title2d = Text("2-D Matrix A  (3×4)", font_size=26, color=ACCENT_BLUE)
        title2d.next_to(grid2d, UP, buff=0.2)

        self.play(FadeIn(grid2d), FadeIn(title2d))
        self.wait(0.8)

        narration(self,
            "GPU memory is one long flat array — no 2-D structure.",
            duration=2.5)

        # ── 1-D flat strip ────────────────────────────────────────────────
        flat = VGroup()
        flat_cells = []
        total = ROWS * COLS
        for idx in range(total):
            r, c = divmod(idx, COLS)
            rect = Rectangle(
                width=cell_sz * 0.85, height=cell_sz * 0.55,
                fill_color=ACCENT_GREEN, fill_opacity=0.25,
                stroke_color=ACCENT_GREEN, stroke_width=1.5
            ).move_to(RIGHT * idx * (cell_sz * 0.85 + 0.04))
            lbl = Text(str(idx), font_size=13, color=ACCENT_GREEN).move_to(rect)
            flat.add(rect, lbl)
            flat_cells.append(rect)
        flat.center().shift(DOWN * 1.5)

        title1d = Text("1-D Memory (row-major)", font_size=26, color=ACCENT_GREEN)
        title1d.next_to(flat, UP, buff=0.2)

        self.play(FadeIn(title1d), FadeIn(flat))

        # ── Animate flattening row-by-row ─────────────────────────────────
        narration(self,
            "Row by row, we lay each row end-to-end.",
            duration=2.5)

        for r in range(ROWS):
            row_rects = VGroup(*[grid_cells[(r, c)] for c in range(COLS)])
            self.play(row_rects.animate.set_fill(ACCENT_GOLD, opacity=0.7),
                      run_time=0.3)
            self.wait(0.25)
            self.play(row_rects.animate.set_fill(ACCENT_BLUE, opacity=0.25),
                      run_time=0.25)

        # ── Highlight A[1][2] → index 1*4+2=6 ────────────────────────────
        narration(self,
            "To access A[1][2]: compute 1×K + 2 = index 6.",
            duration=2.5)

        target_r, target_c = 1, 2
        K = COLS
        flat_idx = target_r * K + target_c

        highlight_2d = glow_rect(grid_cells[(target_r, target_c)], ACCENT_GOLD)
        highlight_1d = glow_rect(flat_cells[flat_idx], ACCENT_GOLD)

        arrow = CurvedArrow(
            grid_cells[(target_r, target_c)].get_bottom(),
            flat_cells[flat_idx].get_top(),
            angle=-0.5, color=ACCENT_GOLD, stroke_width=2
        )

        self.play(Create(highlight_2d), Create(highlight_1d), Create(arrow))

        formula = MathTex(
            r"A[\text{row}][\text{col}] \;\longrightarrow\; A[\text{row} \times K + \text{col}]",
            font_size=32, color=ACCENT_GOLD
        ).to_edge(DOWN).shift(UP * 0.3)

        self.play(Write(formula))
        self.wait(2.5)

        self.play(FadeOut(VGroup(grid2d, title2d, flat, title1d,
                                  highlight_2d, highlight_1d, arrow, formula)))


# ─────────────────────────────────────────────────────────────────────────────
# SCENE 8 — INNER LOOP (single thread computes its element)
# ─────────────────────────────────────────────────────────────────────────────
class S08_InnerLoop(Scene):
    def construct(self):
        self.camera.background_color = BG_COLOR
        chapter_title(self, "The Inner Loop", "One thread's full computation")

        M, K, N = 4, 4, 4
        cell = 0.50

        A_data = np.random.randint(1, 5, (M, K)).tolist()
        B_data = np.random.randint(1, 5, (K, N)).tolist()

        def make_labeled_grid(data, rows, cols, color, matrix_label):
            vg = VGroup()
            cell_objs = []
            for r in range(rows):
                row = []
                for c in range(cols):
                    rect = Rectangle(
                        width=cell, height=cell,
                        fill_color=color, fill_opacity=0.18,
                        stroke_color=color, stroke_width=1.3
                    ).move_to(RIGHT * c * (cell+0.04) + DOWN * r * (cell+0.04))
                    val = Text(str(data[r][c]), font_size=15, color=color).move_to(rect)
                    vg.add(rect, val)
                    row.append(rect)
                cell_objs.append(row)
            vg.center()
            lbl = Text(matrix_label, font_size=24, color=color).next_to(vg, UP, buff=0.15)
            return VGroup(vg, lbl), cell_objs

        A_mob, a_cells = make_labeled_grid(A_data, M, K, ACCENT_BLUE, "A")
        B_mob, b_cells = make_labeled_grid(B_data, K, N, ACCENT_GREEN, "B")

        grp = VGroup(A_mob, B_mob).arrange(RIGHT, buff=1.2).shift(UP * 0.8)
        self.play(FadeIn(grp))

        # The target thread computes C[2][1]
        TARGET_ROW, TARGET_COL = 2, 1

        narration(self,
            f"Our thread is responsible for C[{TARGET_ROW}][{TARGET_COL}].",
            duration=2.5)

        # Show C placeholder
        C_rect = Rectangle(width=cell, height=cell,
                           fill_color=ACCENT_GOLD, fill_opacity=0.8,
                           stroke_color=ACCENT_GOLD, stroke_width=2)
        C_lbl  = Text(f"C[{TARGET_ROW}][{TARGET_COL}]", font_size=18,
                      color=ACCENT_GOLD).next_to(C_rect, DOWN, buff=0.1)
        C_val  = Text("0.0", font_size=14, color=WHITE).move_to(C_rect)
        C_grp  = VGroup(C_rect, C_val, C_lbl).to_edge(RIGHT).shift(LEFT * 1.0 + UP * 0.5)

        self.play(FadeIn(C_grp))

        # Loop annotation
        loop_code = Code(
            code_string="""float temp = 0.0;
for (int i = 0; i < K; ++i) {
    temp += A[row*K + i] * B[i*N + col];
}""",
            language="cpp",
            background="rectangle",
            background_config={"color": ACCENT_TEAL, "stroke_width": 2},
            paragraph_config={"font_size": 22},
            add_line_numbers=False,
        ).to_edge(DOWN).shift(UP * 0.3)
        self.play(FadeIn(loop_code))

        running_total = 0.0
        for k_idx in range(K):
            # Highlight A[TARGET_ROW][k_idx]
            a_h = a_cells[TARGET_ROW][k_idx].copy()
            a_h.set_fill(ACCENT_BLUE, opacity=0.95).set_stroke(ACCENT_GOLD, 2.5)

            # Highlight B[k_idx][TARGET_COL]
            b_h = b_cells[k_idx][TARGET_COL].copy()
            b_h.set_fill(ACCENT_GREEN, opacity=0.95).set_stroke(ACCENT_GOLD, 2.5)

            product = A_data[TARGET_ROW][k_idx] * B_data[k_idx][TARGET_COL]
            running_total += product

            step_txt = Text(
                f"k={k_idx}: {A_data[TARGET_ROW][k_idx]} × {B_data[k_idx][TARGET_COL]}"
                f" = {product}  →  temp = {running_total:.0f}",
                font_size=20, color=DIM_WHITE
            ).to_edge(UP).shift(DOWN * 0.25)

            self.play(
                FadeIn(a_h), FadeIn(b_h), FadeIn(step_txt),
                run_time=0.4
            )
            # Update C display
            C_val_new = Text(f"{running_total:.0f}", font_size=14, color=WHITE).move_to(C_rect)
            self.play(Transform(C_val, C_val_new), run_time=0.25)
            self.wait(0.55)
            self.play(FadeOut(a_h), FadeOut(b_h), FadeOut(step_txt), run_time=0.2)

        narration(self,
            "After K iterations, temp holds the dot product!",
            duration=2.5)

        alpha_eq = MathTex(
            r"C[\text{row}][\text{col}] = \alpha \cdot \text{temp} + \beta \cdot C[\text{row}][\text{col}]",
            font_size=28, color=ACCENT_GOLD
        ).next_to(loop_code, UP, buff=0.3)
        self.play(Write(alpha_eq))
        self.wait(2)
        self.play(FadeOut(VGroup(grp, C_grp, loop_code, alpha_eq)))


# ─────────────────────────────────────────────────────────────────────────────
# SCENE 9 — FULL KERNEL EXECUTION
# ─────────────────────────────────────────────────────────────────────────────
class S09_FullKernel(Scene):
    def construct(self):
        self.camera.background_color = BG_COLOR
        chapter_title(self, "Full Kernel", "All threads run simultaneously")

        SIZE = 6
        cell_sz = 0.58

        # Output matrix C, all empty
        c_cells = {}
        grid_vg = VGroup()
        for r in range(SIZE):
            for c in range(SIZE):
                rect = Rectangle(
                    width=cell_sz, height=cell_sz,
                    fill_color="#111122", fill_opacity=0.95,
                    stroke_color=ACCENT_BLUE, stroke_width=0.7
                ).move_to(RIGHT * c * cell_sz + DOWN * r * cell_sz)
                c_cells[(r, c)] = rect
                grid_vg.add(rect)
        grid_vg.center()

        lbl = Text("Output matrix C", font_size=26, color=DIM_WHITE).next_to(grid_vg, UP, buff=0.2)
        self.play(FadeIn(grid_vg), FadeIn(lbl))
        self.wait(0.5)

        narration(self,
            "Every cell is empty. Now we launch the kernel ...",
            duration=2)

        # Light up all cells simultaneously (simulate parallel execution)
        colors_fill = [ACCENT_BLUE, ACCENT_GREEN, ACCENT_PURPLE,
                       ACCENT_GOLD, ACCENT_TEAL, ACCENT_RED]
        animations = []
        for r in range(SIZE):
            for c in range(SIZE):
                animations.append(
                    c_cells[(r, c)].animate.set_fill(
                        colors_fill[r % len(colors_fill)], opacity=0.55
                    )
                )

        narration(self,
            "All threads fire at once — each filling its unique cell.",
            duration=3)

        self.play(*animations, run_time=1.8)
        self.wait(1)

        # Show thread labels on a few cells
        sample_labels = [
            (0, 0, "T(0,0)"), (0, 5, "T(0,5)"),
            (5, 0, "T(5,0)"), (5, 5, "T(5,5)"),
            (2, 3, "T(2,3)"),
        ]
        label_mobs = VGroup()
        for r, c, txt in sample_labels:
            t = Text(txt, font_size=9, color=WHITE).move_to(c_cells[(r, c)])
            label_mobs.add(t)
        self.play(FadeIn(label_mobs))
        self.wait(2)

        summary = Text("Matrix C is fully computed in one kernel launch!",
                       font_size=30, color=ACCENT_GOLD).next_to(grid_vg, DOWN, buff=0.5)
        self.play(Write(summary))
        self.wait(2.5)
        self.play(FadeOut(VGroup(grid_vg, lbl, label_mobs, summary)))


# ─────────────────────────────────────────────────────────────────────────────
# SCENE 10 — FULL KERNEL CODE WALKTHROUGH
# ─────────────────────────────────────────────────────────────────────────────
class S10_KernelCode(Scene):
    def construct(self):
        self.camera.background_color = BG_COLOR
        chapter_title(self, "The Complete Kernel", "Reading every line")

        full_code = Code(
            code_string="""__global__ void matmul_naive(
    int M, int N, int K, float alpha,
    const float *A, const float *B,
    float beta,  float *C
){
    // 1. Compute global row & column for this thread
    const uint x = blockIdx.x * blockDim.x + threadIdx.x;
    const uint y = blockIdx.y * blockDim.y + threadIdx.y;

    // 2. Bounds check — ignore out-of-matrix threads
    if (x < M && y < N) {
        float temp = 0.0;

        // 3. Dot product: row x of A  ·  col y of B
        for (int i = 0; i < K; ++i) {
            temp += A[x*K + i] * B[i*N + y];
        }

        // 4. Scale & accumulate into C
        C[x*N + y] = alpha * temp + beta * C[x*N + y];
    }
}""",
            language="cpp",
            background="rectangle",
            background_config={"color": ACCENT_BLUE, "stroke_width": 2},
            paragraph_config={"font_size": 19},
            add_line_numbers=True,
        )
        full_code.center().scale_to_fit_height(7.0)

        self.play(FadeIn(full_code))
        self.wait(1)

        # Annotate line groups
        annotations = [
            ("Lines 7-8: Each thread computes its unique (row, col)", ACCENT_GOLD),
            ("Lines 11: Discard threads beyond matrix edge", ACCENT_RED),
            ("Lines 14-16: Dot product inner loop — K multiplications", ACCENT_GREEN),
            ("Line 19: Write result with alpha/beta scaling", ACCENT_TEAL),
        ]

        for txt, color in annotations:
            ann = Text(txt, font_size=22, color=color).to_edge(DOWN).shift(UP * 0.3)
            self.play(FadeIn(ann), run_time=0.5)
            self.wait(2.2)
            self.play(FadeOut(ann), run_time=0.3)

        self.play(FadeOut(full_code))


# ─────────────────────────────────────────────────────────────────────────────
# SCENE 11 — PERFORMANCE DISCUSSION (why it's "naive")
# ─────────────────────────────────────────────────────────────────────────────
class S11_Performance(Scene):
    def construct(self):
        self.camera.background_color = BG_COLOR
        chapter_title(self, "Why 'Naive'?", "The performance bottleneck")

        # Memory access diagram
        header = Text("Problem: Global Memory Access", font_size=32,
                      color=ACCENT_RED).to_edge(UP).shift(DOWN * 0.4)
        self.play(FadeIn(header))

        # Draw two threads both accessing same row of A
        def draw_mem_strip(n, label, color, y_pos):
            strip = VGroup()
            for i in range(n):
                rect = Rectangle(width=0.45, height=0.35,
                                 fill_color=color, fill_opacity=0.25,
                                 stroke_color=color, stroke_width=1.2)
                rect.move_to(RIGHT * i * 0.48 + UP * y_pos)
                strip.add(rect)
            strip.center()
            lbl = Text(label, font_size=20, color=color)
            lbl.next_to(strip, LEFT, buff=0.25)
            return VGroup(strip, lbl), strip

        a_strip_vg, a_strip = draw_mem_strip(8, "A (Global Memory)", ACCENT_BLUE, 1.5)
        b_strip_vg, b_strip = draw_mem_strip(8, "B (Global Memory)", ACCENT_GREEN, 0.5)

        self.play(FadeIn(a_strip_vg), FadeIn(b_strip_vg))

        # Two thread arrows showing repeated reads
        arrows = VGroup()
        for tx in range(3):
            arr = Arrow(
                a_strip[tx].get_bottom() + DOWN * 0.05,
                a_strip[tx].get_bottom() + DOWN * 0.6,
                color=ACCENT_RED, stroke_width=2, buff=0
            )
            lbl = Text(f"T{tx}", font_size=13, color=ACCENT_RED)
            lbl.next_to(arr, DOWN, buff=0.05)
            arrows.add(arr, lbl)

        self.play(Create(arrows), run_time=0.8)

        narration(self,
            "Every thread reads the SAME rows/cols from slow global memory.",
            duration=3)

        problem_list = VGroup(
            Text("✗  No caching — repeated DRAM reads", font_size=24, color=ACCENT_RED),
            Text("✗  No shared memory reuse between threads", font_size=24, color=ACCENT_RED),
            Text("✗  Each element of A/B read M or N times!", font_size=24, color=ACCENT_RED),
        ).arrange(DOWN, buff=0.4, aligned_edge=LEFT).shift(DOWN * 1.2)

        for item in problem_list:
            self.play(FadeIn(item, shift=RIGHT * 0.2), run_time=0.5)
            self.wait(0.8)

        self.wait(1)

        solution = Text("→  Tiled / shared-memory GEMM can be 10–30× faster",
                        font_size=26, color=ACCENT_GREEN).next_to(problem_list, DOWN, buff=0.5)
        self.play(Write(solution))
        self.wait(2.5)

        self.play(FadeOut(VGroup(header, a_strip_vg, b_strip_vg,
                                  arrows, problem_list, solution)))


# ─────────────────────────────────────────────────────────────────────────────
# SCENE 12 — CLOSING SUMMARY
# ─────────────────────────────────────────────────────────────────────────────
class S12_Closing(Scene):
    def construct(self):
        self.camera.background_color = BG_COLOR

        title = Text("What We Learned", font_size=48, color=ACCENT_GOLD,
                     weight=BOLD).to_edge(UP).shift(DOWN * 0.5)
        self.play(Write(title))

        points = [
            ("🧮", "Matrix multiplication: row·column dot products", ACCENT_BLUE),
            ("🏢", "Grid → Blocks → Threads (apartment analogy)", ACCENT_GREEN),
            ("📍", "global_xy = blockIdx × blockDim + threadIdx", ACCENT_GOLD),
            ("📏", "Ceiling division + bounds check for any size", ACCENT_RED),
            ("🗂️",  "Row-major flat index: A[r][c] = A[r*K+c]", ACCENT_PURPLE),
            ("🔄", "Inner loop: K multiplications per thread", ACCENT_TEAL),
            ("⚡", "All threads run in parallel → fills C at once", DIM_WHITE),
            ("🐢", "'Naive' = no shared memory → use tiling next!", ACCENT_RED),
        ]

        items = VGroup()
        for emoji, text, color in points:
            line = Text(f"{emoji}  {text}", font_size=22, color=color)
            items.add(line)
        items.arrange(DOWN, buff=0.32, aligned_edge=LEFT)
        items.next_to(title, DOWN, buff=0.4).shift(LEFT * 0.5)

        for item in items:
            self.play(FadeIn(item, shift=UP * 0.1), run_time=0.3)
            self.wait(0.25)

        self.wait(2)

        farewell = Text("Happy GPU programming! 🚀",
                        font_size=36, color=ACCENT_GOLD, weight=BOLD).to_edge(DOWN)
        self.play(Write(farewell))
        self.wait(3)
        self.play(FadeOut(VGroup(title, items, farewell)))


# ─────────────────────────────────────────────────────────────────────────────
# COMBINED SCENE  (renders all chapters in sequence)
# ─────────────────────────────────────────────────────────────────────────────
class CUDAGEMMAnimation(Scene):
    """
    Master scene — runs all chapters back-to-back.
    Render with:
        manim -pql cuda_gemm_animation.py CUDAGEMMAnimation
    """
    def construct(self):
        scenes = [
            S01_Introduction,
            S02_MatMulIntuition,
            S03_ApartmentAnalogy,
            S04_BlocksAndGrids,
            S05_ThreadMapping,
            S06_CeilingDivision,
            S07_MemoryFlattening,
            S08_InnerLoop,
            S09_FullKernel,
            S10_KernelCode,
            S11_Performance,
            S12_Closing,
        ]
        for SceneClass in scenes:
            SceneClass.construct(self)
            # Brief black pause between chapters
            self.wait(0.3)


# ─────────────────────────────────────────────────────────────────────────────
# INDIVIDUAL SCENE RUNNERS (useful for testing one chapter at a time)
# ─────────────────────────────────────────────────────────────────────────────
# Run a single chapter:
#   manim -pql cuda_gemm_animation.py S03_ApartmentAnalogy
