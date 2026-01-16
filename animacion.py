from manim import *
import numpy as np
from regresion_sinumbral import derivada,train,h
class Regresion_NoLineal(Scene):
    def construct(self):
        # Configurar ejes
        axes = Axes(
            x_range=[0, 10, 1],
            y_range=[0, 250, 50],
            x_length=7,
            y_length=5,
            axis_config={"include_tip": True}
        ).to_edge(LEFT)
        
        # Etiquetas de ejes
        x_label = Text("x", font_size=24).next_to(axes.x_axis.get_end(), DOWN)
        y_label = Text("y", font_size=24).next_to(axes.y_axis.get_end(), LEFT)
        
        self.play(Create(axes), Write(x_label), Write(y_label))
        
        # Generar datos
        np.random.seed(42)
        x_data = np.random.uniform(0, 10, 30)
        y_data = 2*x_data**2 + 3*x_data + 1 + np.random.normal(0, 20, len(x_data))
        
        # Puntos de datos
        dots = VGroup(*[
            Dot(axes.c2p(x_data[i], y_data[i]), color=BLUE, radius=0.08)
            for i in range(len(x_data))
        ])
        self.play(Create(dots))
        
        # Título
        title = Text("Regresion No Lineal", font_size=32).to_edge(UP)
        self.play(Write(title))
        
        # Panel de información inicial
        info_panel = VGroup(
            Text("Iteracion: 0", font_size=24),
            Text("Error: ---", font_size=24),
            Text("w0: ---", font_size=20),
            Text("w1: ---", font_size=20),
            Text("w2: ---", font_size=20),
        ).arrange(DOWN, aligned_edge=LEFT).to_edge(RIGHT).shift(UP*0.5)
        
        self.play(Write(info_panel))
        
        # usanfo la función train
        w_final, error_final, historial = train(
            x_data, y_data,
            alfa=0.0001,
            p=2,
            guardar_cada=500
        )
        
        # Preparar valores para graficar la curva
        x_vals = np.linspace(0, 10, 100)
        
        # Curva inicial (primer estado)
        y_pred_inicial = h(x_vals, historial[0]['w'])
        curve = axes.plot_line_graph(
            x_values=x_vals,
            y_values=y_pred_inicial,
            line_color=RED,
            add_vertex_dots=False
        )
        self.play(Create(curve))
        
        # Animar cada estado del historial de train que almacenamos
        for estado in historial[1:]:  # Desde el segundo estado
            y_new = h(x_vals, estado['w'])
            new_curve = axes.plot_line_graph(
                x_values=x_vals,
                y_values=y_new,
                line_color=RED, #Definimos el color de la curva
                add_vertex_dots=False
            )
            
            w = estado['w']
            new_info = VGroup(
                Text(f"Iteracion: {estado['iter']}", font_size=24),
                Text(f"Error: {estado['error']:.4f}", font_size=24),
                Text(f"w0: {w[0]:.3f}", font_size=20),
                Text(f"w1: {w[1]:.3f}", font_size=20),
                Text(f"w2: {w[2]:.3f}", font_size=20),
            ).arrange(DOWN, aligned_edge=LEFT).to_edge(RIGHT).shift(UP*0.5)
            
            self.play(
                Transform(curve, new_curve),
                Transform(info_panel, new_info),
                run_time=1
            )
        
        # Ecuación final
        final_eq = Text(
            f"y = {w_final[2]:.2f}x² + {w_final[1]:.2f}x + {w_final[0]:.2f}",
            color=YELLOW, #Definimos el color de la función
            font_size=28
        ).next_to(title, DOWN)
        
        real_eq = Text(
            "Real: y = 2x² + 3x + 1",
            color=GREEN, #Defino el color de la ecuación
            font_size=24
        ).next_to(final_eq, DOWN)
        
        self.play(Write(final_eq), Write(real_eq))
        self.wait(3)
