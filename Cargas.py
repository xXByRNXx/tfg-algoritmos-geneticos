import numpy as np
import pygad
import matplotlib.pyplot as plt
from scipy.spatial.distance import pdist

# Configuración del problema
N_CHARGES = 6          # Número de cargas eléctricas
BOX_SIZE = 5.0          # La caja irá de -5 a 5 en los ejes X e Y
GENERATIONS = 150       # Número de generaciones del algoritmo

# 1. Configurar la figura 
plt.ion()  # Activar modo interactivo
fig, ax = plt.subplots(figsize=(6, 6))
ax.set_xlim(-BOX_SIZE - 1, BOX_SIZE + 1)
ax.set_ylim(-BOX_SIZE - 1, BOX_SIZE + 1)
ax.grid(True, linestyle='--', alpha=0.6)

# Dibujar los límites de la caja
rect = plt.Rectangle((-BOX_SIZE, -BOX_SIZE), BOX_SIZE*2, BOX_SIZE*2, 
                     fill=False, color='black', linewidth=2)
ax.add_patch(rect)

# Crear los elementos gráficos que se actualizarán
scatter = ax.scatter([], [], c='red', edgecolors='black', s=100, zorder=5)
titulo = ax.set_title("Inicializando sistema...", fontsize=12)
texto_energia = ax.text(-BOX_SIZE, -BOX_SIZE - 0.7, "", fontsize=10, color='darkblue')

def fitness_func(ga_instance, solution, solution_idx):
    """Calcula la aptitud de la solución."""
    posiciones = np.reshape(solution, (N_CHARGES, 2))
    distancias = pdist(posiciones)
    distancias[distancias < 0.001] = 0.001
    energia = np.sum(1.0 / distancias)
    return -energia

def on_generation(ga_instance):
    """Callback que actualiza el gráfico en vivo en cada generación."""
    best_solution, best_fitness, _ = ga_instance.best_solution()
    posiciones = np.reshape(best_solution, (N_CHARGES, 2))
    
    # 1. Actualizar las posiciones de las cargas
    scatter.set_offsets(posiciones)
    
    # 2. Actualizar la información en texto
    gen = ga_instance.generations_completed
    energia_actual = -best_fitness
    titulo.set_text(f"Generación: {gen} / {GENERATIONS}")
    texto_energia.set_text(f"Energía potencial del sistema: {energia_actual:.4f}")
    
    # 3. Forzar el refresco de la ventana de Matplotlib
    fig.canvas.draw()
    fig.canvas.flush_events()
    plt.pause(0.01)  # Pausa mínima para que el sistema operativo pinte el frame

# 2. Configuración de PyGAD
espacio_genes = {'low': -BOX_SIZE, 'high': BOX_SIZE}

ga_instance = pygad.GA(
    num_generations=GENERATIONS,
    num_parents_mating=20,
    fitness_func=fitness_func,
    sol_per_pop=100,
    num_genes=N_CHARGES * 2,
    gene_space=espacio_genes,
    mutation_type="random",
    mutation_probability=0.15,
    on_generation=on_generation
)

# 3. Ejecutar el algoritmo
print("Iniciando evolución del sistema. Observa la ventana gráfica...")
ga_instance.run()

# 4. Finalizar
plt.ioff()  # Desactivar el modo interactivo
plt.show()  # Mantener la ventana abierta al finalizar el proceso