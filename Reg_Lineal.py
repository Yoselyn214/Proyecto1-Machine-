from manim import *
import numpy as np

class IntroRegression(Scene):
    def construct(self):
        font_main = "Arial"

        # =========================
        # 1) INTRO (pantalla inicial)
        # =========================
        titulo = Text("Regression", font_size=72, font=font_main)
        subtitulo = Text("Linear vs Nonlinear", font_size=32, font=font_main)
        subtitulo.next_to(titulo, DOWN, buff=0.2)

        autor = Text("Group 10 \n" \
        "Diego Lopez\n" \
        "Yoselin Miranda", font_size=20, font=font_main)
        autor.to_edge(DOWN, buff=0.5)

        self.play(Write(titulo))
        self.play(FadeIn(subtitulo))
        self.play(FadeIn(autor))
        self.wait(2.5)

        # =========================
        # 2) "Linear" sale del subtitulo y sube 
        # =========================
        linear = Text("Linear", font_size=32, font=font_main)

        self.play(FadeOut(titulo), FadeOut(autor), run_time=0.6)

        linear.move_to(subtitulo.get_left() + RIGHT * (linear.width / 2 + 0.5))

        self.play(
            linear.animate.scale(1.6).move_to(UP * 3.1).set_color(BLUE),
            FadeOut(subtitulo),
            run_time=1.0
        )

        # =========================
        # 3) GRÁFICOS: ejes + puntos + recta
        # =========================
        axes = Axes(
            x_range=[0, 10, 1],
            y_range=[0, 10, 1],
            x_length=10,
            y_length=6,
            tips=False
        ).to_edge(DOWN, buff=0.9)

        # Asegura que los ejes queden por debajo de la recta/puntos
        axes.set_z_index(0)

        self.play(Create(axes), run_time=1.0)

        # Puntos "al azar" con tendencia
        rng = np.random.default_rng(10)
        xs = np.linspace(1, 9, 30)
        ys = 0.65 * xs + 1.8 + rng.normal(0, 1.2 , size=len(xs))


        dots = VGroup(*[
    Dot(
        axes.c2p(x, y),
        radius=0.12,         
        color=GREEN
        ).set_z_index(3)
    for x, y in zip(xs, ys)])


        self.play(LaggedStartMap(FadeIn, dots, lag_ratio=0.08), run_time=1.2)
        self.wait(5.0)

        # Recta inicial (ENCIMA)
        m0, b0 = 0.9, 0.5
        line = axes.plot(lambda x: m0 * x + b0, x_range=[0, 10])
        line.set_z_index(3)              
        line.set_stroke(width=6)         

        self.play(Create(line), run_time=0.9)
        self.wait(1.5)

        # La recta se mueve arriba/abajo
        offsets = [2.8, 1.8, 0.8, 0, -0.8]
        for off in offsets:
            new_line = axes.plot(lambda x, m=m0, b=off: m * x + b, x_range=[0, 10])
            new_line.set_z_index(3)
            new_line.set_stroke(width=6)
            self.play(Transform(line, new_line), run_time=0.8)

        self.wait(0.6)

        # Regresión lineal final (mínimos cuadrados)
        m1, b1 = np.polyfit(xs, ys, 1)
        best_line = axes.plot(lambda x: m1 * x + b1, x_range=[0, 10])
        best_line.set_z_index(3)
        best_line.set_stroke(width=6)

        self.play(Transform(line, best_line), run_time=1.0)
        self.wait(1.0)

        # =========================
        # 4) ECUACIÓN MODELO y luego FUNCIÓN DE PÉRDIDA 
        # =========================
        model_eq = MathTex(r"h(x_i) = b + w x_i").scale(1.1)
        model_eq.to_edge(DOWN, buff=0.4).shift(UP * 1.3 + RIGHT * 2.2)

        self.play(Write(model_eq), run_time=1.0)
        self.wait(15.0)

        # =========================
        # 5) LÍNEAS ROJAS DE ERROR 
        # =========================
        error_lines = VGroup(*[ Line(
        start=axes.c2p(x, y),
        end=axes.c2p(x, m1 * x + b1),
        stroke_width=6,      
        color=RED,
        stroke_opacity=0.9
    ).set_z_index(2)
    for x, y in zip(xs, ys)])


        self.play(LaggedStartMap(Create, error_lines, lag_ratio=0.08), run_time=1.2)
        self.wait(0.8)

        loss_eq = MathTex(
            r"\mathcal{L}(b,w)=\frac{1}{n}\sum_{i=1}^{n}(h(x_i)-y_i)^2"
        ).scale(0.85)

        loss_eq.move_to(model_eq)

        self.play(Transform(model_eq, loss_eq), run_time=1.1)
        self.wait(12)