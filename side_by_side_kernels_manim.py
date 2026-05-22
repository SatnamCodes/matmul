"""
Matrix-first Manim explainer for the two CUDA GEMM kernels in this repo.

Render:
    .venv/bin/manim -pql side_by_side_kernels_manim.py SideBySideKernels

High quality:
    .venv/bin/manim -pqh side_by_side_kernels_manim.py SideBySideKernels
"""

from manim import *


config.background_color = "#0b1020"

BG = "#0b1020"
PANEL = "#111827"
PANEL_DARK = "#020617"
TEXT = "#e5e7eb"
MUTED = "#9ca3af"
A_BLUE = "#38bdf8"
B_GREEN = "#34d399"
C_GOLD = "#fbbf24"
NAIVE_RED = "#fb7185"
COAL_PURPLE = "#a78bfa"
WHITE_SOFT = "#f8fafc"


def txt(value, size=24, color=TEXT, weight=NORMAL):
    return Text(value, font_size=size, color=color, weight=weight)


def make_caption(value, color=TEXT):
    box = RoundedRectangle(
        width=12.7,
        height=0.58,
        corner_radius=0.08,
        fill_color=PANEL_DARK,
        fill_opacity=0.95,
        stroke_color="#334155",
        stroke_width=1,
    ).to_edge(DOWN, buff=0.14)
    line = txt(value, 21, color, BOLD).move_to(box)
    return VGroup(box, line)


def make_panel(title, subtitle, stroke_color):
    box = RoundedRectangle(
        width=6.55,
        height=5.85,
        corner_radius=0.12,
        fill_color=PANEL,
        fill_opacity=0.96,
        stroke_color=stroke_color,
        stroke_width=2,
    )
    heading = txt(title, 29, stroke_color, BOLD).move_to(box.get_top() + DOWN * 0.38)
    sub = txt(subtitle, 17, MUTED).next_to(heading, DOWN, buff=0.06)
    return VGroup(box, heading, sub)


def make_matrix(rows, cols, name, color, cell_size=0.18):
    grid = VGroup()
    cells = []
    for r in range(rows):
        row = []
        for c in range(cols):
            rect = Square(
                side_length=cell_size,
                fill_color="#172033",
                fill_opacity=1,
                stroke_color="#475569",
                stroke_width=0.75,
            )
            rect.move_to(
                RIGHT * c * (cell_size + 0.018)
                + DOWN * r * (cell_size + 0.018)
            )
            grid.add(rect)
            row.append(rect)
        cells.append(row)
    grid.center()
    border = SurroundingRectangle(grid, color=color, buff=0.035, stroke_width=1.25)
    name_mob = txt(name, 16, color, BOLD).next_to(border, UP, buff=0.07)
    return VGroup(grid, border, name_mob), cells


def make_kernel_matrices():
    a_group, a_cells = make_matrix(6, 4, "A", A_BLUE)
    b_group, b_cells = make_matrix(4, 6, "B", B_GREEN)
    c_group, c_cells = make_matrix(6, 6, "C", C_GOLD)
    times = txt("x", 18, MUTED)
    equals = txt("=", 18, MUTED)
    mats = VGroup(a_group, times, b_group, equals, c_group).arrange(RIGHT, buff=0.13)
    return mats, a_cells, b_cells, c_cells


def overlay(cell, color, opacity=0.82):
    copy = cell.copy()
    copy.set_fill(color, opacity=opacity)
    copy.set_stroke(color, width=2)
    return copy


def row_highlight(cells, row, color, opacity=0.72):
    return VGroup(*[overlay(cell, color, opacity) for cell in cells[row]])


def col_highlight(cells, col, color, opacity=0.72):
    return VGroup(*[overlay(row[col], color, opacity) for row in cells])


def cells_highlight(cells, positions, color, opacity=0.86):
    return VGroup(*[overlay(cells[r][c], color, opacity) for r, c in positions])


def lane_markers(cells, positions, color):
    markers = VGroup()
    for lane, (r, c) in enumerate(positions):
        dot = Circle(
            radius=0.064,
            fill_color=color,
            fill_opacity=1,
            stroke_color=WHITE_SOFT,
            stroke_width=0.8,
        ).move_to(cells[r][c])
        number = txt(str(lane), 7, PANEL_DARK, BOLD).move_to(dot)
        markers.add(VGroup(dot, number))
    return markers


def make_memory_strip(color):
    strip = VGroup()
    cells = []
    for i in range(36):
        rect = Rectangle(
            width=0.105,
            height=0.28,
            fill_color="#1f2937",
            fill_opacity=1,
            stroke_color="#475569",
            stroke_width=0.45,
        )
        rect.move_to(RIGHT * i * 0.118)
        strip.add(rect)
        cells.append(rect)
    strip.center()
    border = SurroundingRectangle(strip, color=color, buff=0.04, stroke_width=1)
    title = txt("row-major C memory", 13, MUTED).next_to(border, DOWN, buff=0.06)
    return VGroup(strip, border, title), cells


def strip_highlight(cells, indexes, color):
    return VGroup(*[overlay(cells[i], color, 0.92) for i in indexes])


class SideBySideKernels(Scene):
    def construct(self):
        self.camera.background_color = BG

        title = txt("Two kernels. Same A x B. Different thread layout.", 34, C_GOLD, BOLD)
        title.scale_to_fit_width(12.0)
        title.to_edge(UP, buff=0.23)

        left_panel = make_panel("Naive kernel", "2D block: threadIdx.x becomes row", NAIVE_RED)
        right_panel = make_panel("Coalesced kernel", "1D block: threadIdx.x becomes column", COAL_PURPLE)
        panels = VGroup(left_panel, right_panel).arrange(RIGHT, buff=0.28).shift(DOWN * 0.22)

        left_mats, left_a, left_b, left_c = make_kernel_matrices()
        right_mats, right_a, right_b, right_c = make_kernel_matrices()
        left_mats.move_to(left_panel[0].get_center() + UP * 0.78)
        right_mats.move_to(right_panel[0].get_center() + UP * 0.78)

        caption = make_caption("Both panels multiply the same two matrices: A times B gives C.", TEXT)

        self.play(FadeIn(title, shift=DOWN * 0.14), run_time=0.55)
        self.play(FadeIn(panels), FadeIn(left_mats), FadeIn(right_mats), FadeIn(caption), run_time=0.9)
        self.wait(0.55)

        next_caption = make_caption("One thread computes one C cell by taking a row of A and a column of B.", TEXT)
        self.play(FadeOut(caption), run_time=0.14)
        self.play(FadeIn(next_caption), run_time=0.18)
        caption = next_caption

        left_row = row_highlight(left_a, 2, A_BLUE)
        left_col = col_highlight(left_b, 3, B_GREEN)
        left_out = overlay(left_c[2][3], C_GOLD, 0.95)
        right_row = row_highlight(right_a, 2, A_BLUE)
        right_col = col_highlight(right_b, 3, B_GREEN)
        right_out = overlay(right_c[2][3], C_GOLD, 0.95)

        self.play(
            FadeIn(left_row),
            FadeIn(left_col),
            FadeIn(right_row),
            FadeIn(right_col),
            run_time=0.55,
        )
        self.play(FadeIn(left_out), FadeIn(right_out), run_time=0.35)
        pulse_left = SurroundingRectangle(left_out, color=C_GOLD, buff=0.02, stroke_width=3)
        pulse_right = SurroundingRectangle(right_out, color=C_GOLD, buff=0.02, stroke_width=3)
        self.play(Create(pulse_left), Create(pulse_right), run_time=0.4)
        self.wait(0.85)
        self.play(
            FadeOut(VGroup(left_row, left_col, left_out, pulse_left)),
            FadeOut(VGroup(right_row, right_col, right_out, pulse_right)),
            run_time=0.45,
        )

        next_caption = make_caption("Now watch one warp: the math stays the same, but its lanes land on C differently.", TEXT)
        self.play(FadeOut(caption), run_time=0.14)
        self.play(FadeIn(next_caption), run_time=0.18)
        caption = next_caption

        naive_positions = [(r, 2) for r in range(6)]
        coal_positions = [(2, c) for c in range(6)]
        naive_c = cells_highlight(left_c, naive_positions, NAIVE_RED)
        coal_c = cells_highlight(right_c, coal_positions, COAL_PURPLE)
        naive_lanes = lane_markers(left_c, naive_positions, NAIVE_RED)
        coal_lanes = lane_markers(right_c, coal_positions, COAL_PURPLE)

        naive_note = VGroup(
            txt("lane 0,1,2... move down rows", 16, NAIVE_RED, BOLD),
            txt("C writes are separated by N", 14, MUTED),
        ).arrange(DOWN, buff=0.05)
        naive_note.move_to(left_panel[0].get_center() + DOWN * 0.75)

        coal_note = VGroup(
            txt("lane 0,1,2... move across columns", 16, COAL_PURPLE, BOLD),
            txt("C writes are neighbors", 14, MUTED),
        ).arrange(DOWN, buff=0.05)
        coal_note.move_to(right_panel[0].get_center() + DOWN * 0.75)

        self.play(FadeIn(naive_c), FadeIn(coal_c), run_time=0.45)
        self.play(FadeIn(naive_lanes), FadeIn(coal_lanes), FadeIn(naive_note), FadeIn(coal_note), run_time=0.65)
        self.wait(1.1)

        next_caption = make_caption("Naive: lanes use different A rows but mostly the same B column. That is strided memory.", NAIVE_RED)
        self.play(FadeOut(caption), run_time=0.14)
        self.play(FadeIn(next_caption), run_time=0.18)
        caption = next_caption

        naive_a_rows = VGroup(*[row_highlight(left_a, r, NAIVE_RED, 0.42) for r in range(6)])
        naive_b_col = col_highlight(left_b, 2, B_GREEN, 0.85)
        naive_arrow = Arrow(
            left_mats[0].get_bottom() + DOWN * 0.04,
            left_mats[4].get_bottom() + DOWN * 0.04,
            color=NAIVE_RED,
            buff=0.12,
            stroke_width=3,
        )
        self.play(FadeIn(naive_a_rows), FadeIn(naive_b_col), GrowArrow(naive_arrow), run_time=0.75)
        self.wait(0.9)

        next_caption = make_caption("Coalesced: lanes share one A row and walk through neighboring B columns.", COAL_PURPLE)
        self.play(FadeOut(caption), run_time=0.14)
        self.play(FadeIn(next_caption), run_time=0.18)
        caption = next_caption

        coal_a_row = row_highlight(right_a, 2, A_BLUE, 0.85)
        coal_b_cols = VGroup(*[col_highlight(right_b, c, COAL_PURPLE, 0.38) for c in range(6)])
        coal_arrow = Arrow(
            right_mats[0].get_bottom() + DOWN * 0.04,
            right_mats[4].get_bottom() + DOWN * 0.04,
            color=COAL_PURPLE,
            buff=0.12,
            stroke_width=3,
        )
        self.play(FadeIn(coal_a_row), FadeIn(coal_b_cols), GrowArrow(coal_arrow), run_time=0.75)
        self.wait(0.9)

        next_caption = make_caption("Because C is row-major, moving across a row means neighboring memory addresses.", C_GOLD)
        self.play(FadeOut(caption), run_time=0.14)
        self.play(FadeIn(next_caption), run_time=0.18)
        caption = next_caption

        left_mem, left_mem_cells = make_memory_strip(NAIVE_RED)
        right_mem, right_mem_cells = make_memory_strip(COAL_PURPLE)
        left_mem.scale(0.93).move_to(left_panel[0].get_center() + DOWN * 2.1)
        right_mem.scale(0.93).move_to(right_panel[0].get_center() + DOWN * 2.1)
        scattered = strip_highlight(left_mem_cells, [2, 8, 14, 20, 26, 32], NAIVE_RED)
        scattered.move_to(left_mem[0])
        consecutive = strip_highlight(right_mem_cells, list(range(12, 18)), COAL_PURPLE)
        consecutive.move_to(right_mem[0])
        scattered_label = txt("strided", 18, NAIVE_RED, BOLD).next_to(left_mem, UP, buff=0.06)
        consecutive_label = txt("coalesced", 18, COAL_PURPLE, BOLD).next_to(right_mem, UP, buff=0.06)

        self.play(FadeIn(left_mem), FadeIn(right_mem), run_time=0.45)
        self.play(
            FadeIn(scattered),
            FadeIn(consecutive),
            FadeIn(scattered_label),
            FadeIn(consecutive_label),
            run_time=0.65,
        )
        self.wait(1.0)

        next_caption = make_caption("Same dot product per thread. Better lane layout makes the coalesced kernel much faster.", C_GOLD)
        self.play(FadeOut(caption), run_time=0.14)
        self.play(FadeIn(next_caption), run_time=0.18)
        caption = next_caption

        naive_time = txt("13.31 ms avg", 25, NAIVE_RED, BOLD).next_to(naive_note, DOWN, buff=0.2)
        coal_time = txt("2.86 ms avg", 25, COAL_PURPLE, BOLD).next_to(coal_note, DOWN, buff=0.2)
        rule = txt("Rule of thumb: neighboring threads should touch neighboring addresses.", 29, C_GOLD, BOLD)
        rule.move_to(ORIGIN + DOWN * 0.05)

        self.play(FadeIn(naive_time), FadeIn(coal_time), run_time=0.55)
        self.wait(0.7)
        self.play(
            FadeOut(VGroup(
                panels,
                left_mats,
                right_mats,
                naive_c,
                coal_c,
                naive_lanes,
                coal_lanes,
                naive_note,
                coal_note,
                naive_a_rows,
                naive_b_col,
                naive_arrow,
                coal_a_row,
                coal_b_cols,
                coal_arrow,
                left_mem,
                right_mem,
                scattered,
                consecutive,
                scattered_label,
                consecutive_label,
                naive_time,
                coal_time,
                caption,
            )),
            title.animate.scale(0.72).to_edge(UP, buff=0.48),
            run_time=0.8,
        )
        self.play(Write(rule), run_time=0.75)
        self.wait(1.4)
        self.play(FadeOut(VGroup(title, rule)), run_time=0.45)
