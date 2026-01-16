import numpy as np

# HIPOTESIS POLINOMICO
def h(x, w):
    #Hipótesis: polinomio de grado p
    y_pred = np.zeros(len(x))
    for j in range(len(w)):
        y_pred += w[j] * (x ** j)
    return y_pred

# FUNCION DE ERROR MSE
def error(y, y_pred, n):
    L = np.sum((y - y_pred)**2) / (2*n)
    return L

# FUNCIÓN DE DERIVADA
def derivada(x, y, w):
    """Derivada de L respecto a cada peso"""
    n = len(x)
    y_pred = h(x, w)
    dw = []
    
    for j in range(len(w)):
        # Derivada parcial respecto a wj
        gradient = (1/n) * np.sum((y_pred - y) * (x ** j))
        dw.append(gradient)
    
    return np.array(dw)

# ACTUALIZACION DE W
def update(w, dw, alfa):
    """Actualización de pesos con descenso de gradiente"""
    w_nuevo = w - alfa * dw
    return w_nuevo

# ENTRENAMIENTO
def train(x, y, alfa, p, guardar_cada=500):
    """Versión de train que guarda el historial para animación"""
    np.random.seed(2001)
    n = len(x)
    w = np.random.rand(p+1)
    
    y_pred = h(x, w)
    L = error(y, y_pred, n)
    
    iteraciones = 0
    max_iter = 10000
    historial = []
    
    # Guardar estado inicial
    historial.append({
        'iter': 0,
        'w': w.copy(),
        'error': L
    })
    
    #Recorremos con iteracciones:
    for i in range(max_iter):
        dw = derivada(x, y, w)
        w = update(w, dw, alfa)
        y_pred = h(x, w)
        L = error(y, y_pred, n)
        iteraciones = i
        
        # Guardar cada cierto número de iteraciones
        if iteraciones % guardar_cada == 0:
            historial.append({
                'iter': iteraciones,
                'w': w.copy(),
                'error': L
            })
        
        if iteraciones % 1000 == 0:
            print(f"Iteración {iteraciones}, Error: {L}")
    
    # Guardar estado final si no coincide con múltiplo
    if iteraciones % guardar_cada != 0:
        historial.append({
            'iter': iteraciones,
            'w': w.copy(),
            'error': L
        })
    
    return w, L, historial

def main():
    # Generar datos
    x = np.random.uniform(0, 10, 30)
    y = 2*x**2 + 3*x + 1 + np.random.normal(0, 15, len(x))
    
    # Entrenar modelo polinomial de grado 2
    w, error_final, _ = train(x, y, alfa=0.0001, p=2, guardar_cada=500)
    
    print(f"\nPesos encontrados: {w}")
    print(f"Error final: {error_final}")
    print(f"\nPesos reales: [2, 3, 1]")

if __name__ == "__main__":
    main()