from manim import *
import numpy as np
from regresion_sinumbral import h, train, error


class RegresionNoLineal(Scene):
    def construct(self):

        # Definiendo ejes para la gráfica
        axes = Axes(
            x_range=[0, 10, 1],
            y_range=[0, 250, 50],
            x_length=7,
            y_length=5,
            axis_config={"include_tip": True}
        ).to_edge(LEFT)

        #Defino los labels de los ejes
        x_label = Text("x", font_size=24).next_to(axes.x_axis.get_end(), DOWN)
        y_label = Text("y", font_size=24).next_to(axes.y_axis.get_end(), LEFT)

        #Dibujo con manim
        self.play(Create(axes), Write(x_label), Write(y_label))

        # formula de la hipotesis polinomica
        base = Text("h(x) = w0 + w1 x + w2 x", font_size=28)

        pow2 = Text("2", font_size=18)
        pow2.next_to(base[-1], UP, buff=0.05).shift(RIGHT * 0.05)

        dots = Text(" + ... + wp x", font_size=28)

        p_sup = Text("p", font_size=18)
        p_sup.next_to(dots[-1], UP, buff=0.05).shift(RIGHT * 0.05)

        formula = VGroup(base, pow2, dots, p_sup)
        formula.arrange(RIGHT, buff=0.1)
        formula.to_edge(UP)

        #Escribiendo la formula
        self.play(Write(formula))

        # puntos que generamos uniformemente y la formula y
        np.random.seed(42)
        x_data = np.random.uniform(0, 10, 30)
        y_data = 2*x_data**2 + 3*x_data + 1 + np.random.normal(0, 10, len(x_data))

        dots_data = VGroup(*[
            Dot(axes.c2p(x_data[i], y_data[i]), color=BLUE, radius=0.08)
            for i in range(len(x_data))
        ])
        #Graficamos
        self.play(Create(dots_data))

        # entrenando con nuestro modelo train que hicimos en nuestro script
        w_final, error_final, historial = train(
            x_data,
            y_data,
            alfa=0.0001,
            p=2,
            guardar_cada=500
        )

        x_vals = np.linspace(0, 10, 200)

        # curva inicial 
        y_init = h(x_vals, historial[0]["w"])
        curve = axes.plot_line_graph(
            x_values=x_vals,
            y_values=y_init,
            line_color=RED,
            add_vertex_dots=False
        )
        self.play(Create(curve))

        # curva de perdida
        w0_fixed = w_final[0]
        w2_fixed = w_final[2]

        w1_vals = np.linspace(0, 6, 200)

        #Valores de los errores
        loss_vals = []
        for w1 in w1_vals:
            e = error(y_data,h(x_data, [w0_fixed, w1, w2_fixed]),len(x_data)) / len(x_data)
            loss_vals.append(e)



        loss_axes = Axes(
            x_range=[0, 6, 1],
            y_range=[0, max(loss_vals)*1.1, max(loss_vals)/5],
            x_length=5,
            y_length=3,
            axis_config={"include_tip": True}
        ).to_edge(RIGHT)

        loss_curve = loss_axes.plot_line_graph(
            x_values=w1_vals,
            y_values=loss_vals,
            line_color=GREEN,
            add_vertex_dots=False
        )

        self.play(Create(loss_axes), Create(loss_curve))

        # punto que recorre la gráfica con respecto al cambio de w1
        w1_start = historial[0]["w"][1]

        loss_start = error( y_data,h(x_data, [w0_fixed, w1_start, w2_fixed]),len(x_data)) / len(x_data)

        loss_dot = Dot(
            loss_axes.c2p(w1_start, loss_start),
            color=BLUE
        )
        self.add(loss_dot)

        # Animamos el entrenamiento 
        for estado in historial[1:]:
            y_pred = h(x_vals, estado["w"])
            now_curve = axes.plot_line_graph(
                x_values=x_vals,
                y_values=y_pred,
                line_color=RED,
                add_vertex_dots=False
            )
            self.play(Transform(curve, now_curve), run_time=1)

            w1_current = estado["w"][1]
            loss_current = error(y_data,h(x_data, [w0_fixed, w1_current, w2_fixed]),len(x_data)) / len(x_data)

            new_dot = Dot(
                loss_axes.c2p(w1_current, loss_current),
                color=BLUE
            )
            self.play(Transform(loss_dot, new_dot), run_time=0.1)

        # Escribimos la formula de pérdida
        loss_formula = Text(
            "L = (1 / 2n) Σ (y i − h(x i))²",
            font_size=26
        )

        loss_formula.next_to(loss_axes, DOWN, buff=0.4)
        self.play(Write(loss_formula))

        self.wait(3)
