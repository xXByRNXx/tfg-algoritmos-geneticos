import time
import matplotlib.pyplot as plt
import numpy as np
import pygad
from scipy.spatial.distance import pdist

# =====================================================================
# 1. PARÁMETROS FÍSICOS DEL SISTEMA
# =====================================================================
N_CARGAS = 35          # Número de cargas puntuales idénticas
RADIO_DISCO = 1.0      # Radio R del disco conductor (unidades adimensionales)
EPSILON = 1e-6         # Parámetro de regularización de Plummer (evita r_ij = 0)
SEMILLA = 42           # Semilla para reproducibilidad científica

# =====================================================================
# 2. DEFINICIÓN VECTORIZADA DE LA FUNCIÓN DE APTITUD (FITNESS)
# =====================================================================
def calcular_energia_potencial(solution):
    """
    Transforma el cromosoma polar [r_1, theta_1, ..., r_N, theta_N] a coordenadas
    cartesianas 2D y calcula la energía electrostática de forma 100% vectorizada.
    """
    polares = solution.reshape((N_CARGAS, 2))
    r = polares[:, 0]
    theta = polares[:, 1]

    # Transformación vectorizada a cartesianas
    x = r * np.cos(theta)
    y = r * np.sin(theta)
    coordenadas = np.column_stack((x, y))

    # N*(N-1)/2 distancias euclídeas únicas
    distancias = pdist(coordenadas, metric='euclidean')

    # Potencial de Coulomb regularizado (Plummer)
    energia = np.sum(1.0 / np.sqrt(distancias**2 + EPSILON**2))
    return energia

def fitness_func(ga_instance, solution, solution_idx):
    """
    Función de aptitud para PyGAD >= 3.0.
    Maximizar fitness equivale a minimizar energía potencial U.
    """
    energia = calcular_energia_potencial(solution)
    return -energia

# =====================================================================
# 3. CONFIGURACIÓN DEL ESPACIO GENÉTICO Y CALLBACKS
# =====================================================================
# Índices pares: r_i in [0, R] | Índices impares: theta_i in [0, 2*pi]
espacio_genes = []
for _ in range(N_CARGAS):
    espacio_genes.append({'low': 0.0, 'high': RADIO_DISCO})       # r_i
    espacio_genes.append({'low': 0.0, 'high': 2.0 * np.pi})       # theta_i

historial_energia = []

def callback_generacion(ga_instance):
    mejor_solucion, mejor_fitness, _ = ga_instance.best_solution()
    energia_actual = -mejor_fitness
    historial_energia.append(energia_actual)
    
    gen = ga_instance.generations_completed
    if gen % 100 == 0 or gen == 1:
        print(f"Generación {gen:4d} | Energía Mínima U = {energia_actual:.6f}")

# =====================================================================
# 4. INSTANCIACIÓN Y EJECUCIÓN DEL ALGORITMO GENÉTICO (PyGAD)
# =====================================================================
def ejecutar_simulacion():
    print(f"Iniciando optimización evolutiva para N = {N_CARGAS} cargas en un disco de R = {RADIO_DISCO}...")
    t_inicio = time.time()

    ga_instance = pygad.GA(
        num_generations=800,
        num_parents_mating=30,
        fitness_func=fitness_func,
        sol_per_pop=100,
        num_genes=2 * N_CARGAS,
        gene_space=espacio_genes,
        parent_selection_type="tournament",
        K_tournament=4,
        keep_elitism=4,
        crossover_type="uniform",
        crossover_probability=0.85,
        mutation_type="random",
        mutation_by_replacement=False,
        mutation_percent_genes=8,
        random_mutation_min_val=-0.08,
        random_mutation_max_val=0.08,
        on_generation=callback_generacion,
        random_seed=SEMILLA,
        suppress_warnings=True
    )

    ga_instance.run()
    t_total = time.time() - t_inicio

    mejor_solucion, mejor_fitness, _ = ga_instance.best_solution()
    energia_final = -mejor_fitness
    print(f"\nSimulación completada en {t_total:.2f} segundos.")
    print(f"Energía potencial electrostática alcanzada: U_min = {energia_final:.6f}")

    return mejor_solucion, historial_energia

# =====================================================================
# 5. VISUALIZACIÓN FÍSICA Y VALIDACIÓN TEÓRICA
# =====================================================================
def visualizar_resultados(mejor_solucion, historial_energia):
    polares = mejor_solucion.reshape((N_CARGAS, 2))
    r_opt = polares[:, 0]
    theta_opt = polares[:, 1]
    x_opt = r_opt * np.cos(theta_opt)
    y_opt = r_opt * np.sin(theta_opt)

    grid_n = 150
    lim = 1.25 * RADIO_DISCO
    gx = np.linspace(-lim, lim, grid_n)
    gy = np.linspace(-lim, lim, grid_n)
    GX, GY = np.meshgrid(gx, gy)
    
    DX = GX[:, :, np.newaxis] - x_opt[np.newaxis, np.newaxis, :]
    DY = GY[:, :, np.newaxis] - y_opt[np.newaxis, np.newaxis, :]
    DIST_GRID = np.sqrt(DX**2 + DY**2 + 0.03**2)
    V_GRID = np.sum(1.0 / DIST_GRID, axis=2)

    fig, axes = plt.subplots(1, 3, figsize=(18, 5.5))
    plt.subplots_adjust(wspace=0.28)

    # --- PANEL 1: Convergencia ---
    ax1 = axes[0]
    generaciones = np.arange(1, len(historial_energia) + 1)
    ax1.plot(generaciones, historial_energia, color='#1f77b4', linewidth=2)
    ax1.set_title("Convergencia Evolutiva (Teorema de Thomson)", fontsize=12, fontweight='bold')
    ax1.set_xlabel("Generación", fontsize=11)
    ax1.set_ylabel(r"Energía Potencial Electrostática $U$", fontsize=11)
    ax1.grid(True, linestyle='--', alpha=0.6)

    # --- PANEL 2: Mapa Espacial y Cargas ---
    ax2 = axes[1]
    mapa = ax2.contourf(GX, GY, V_GRID, levels=35, cmap='inferno', alpha=0.85)
    fig.colorbar(mapa, ax=ax2, label=r"Potencial Electrostático $V(x,y)$")
    
    circulo = plt.Circle((0, 0), RADIO_DISCO, color='cyan', fill=False, linewidth=2.2, linestyle='--', label='Borde del disco ($r=R$)')
    ax2.add_patch(circulo)
    
    ax2.scatter(x_opt, y_opt, color='white', edgecolors='black', s=75, zorder=5, label=f'Cargas ($N={N_CARGAS}$)')
    ax2.set_title(f"Distribución de Equilibrio en el Disco ($N={N_CARGAS}$)", fontsize=12, fontweight='bold')
    ax2.set_xlabel("$x / R$", fontsize=11)
    ax2.set_ylabel("$y / R$", fontsize=11)
    ax2.set_aspect('equal')
    ax2.set_xlim(-lim, lim)
    ax2.set_ylim(-lim, lim)
    ax2.legend(loc='upper right', fontsize=9)

    # --- PANEL 3: Validación Analítica Continua ---
    ax3 = axes[2]
    r_teorico = np.linspace(0, RADIO_DISCO, 300)
    # Q(r) / Q_total = 1 - sqrt(1 - (r/R)^2)
    q_acumulada_teorica = 1.0 - np.sqrt(1.0 - (r_teorico / RADIO_DISCO)**2)

    r_ordenados = np.sort(r_opt)
    q_acumulada_ga = np.arange(1, N_CARGAS + 1) / N_CARGAS

    ax3.plot(r_teorico, q_acumulada_teorica, 'r--', linewidth=2.2, label=r'Teoría continua: $1 - \sqrt{1 - (r/R)^2}$')
    ax3.step(r_ordenados, q_acumulada_ga, where='post', color='#2ca02c', linewidth=2, label='Resultado AG (Discreto)')
    ax3.scatter(r_ordenados, q_acumulada_ga, color='#2ca02c', s=25, zorder=4)
    ax3.set_title(r"Validación: Fracción de Carga Acumulada $Q(r)/Q_{tot}$", fontsize=12, fontweight='bold')
    ax3.set_xlabel("Radio normalizado $r / R$", fontsize=11)
    ax3.set_ylabel("Fracción de carga encerrada", fontsize=11)
    ax3.grid(True, linestyle='--', alpha=0.6)
    ax3.legend(loc='upper left', fontsize=9.5)

    plt.suptitle("Módulo 1: Minimización Evolutiva de Energía Electrostática en un Disco 2D con PyGAD", fontsize=14, fontweight='bold', y=1.02)
    plt.show()

if __name__ == "__main__":
    solucion_optima, historial = ejecutar_simulacion()
    visualizar_resultados(solucion_optima, historial)