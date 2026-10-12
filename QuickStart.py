import numpy as np
import pygad

# ==============================================================================
# 1. DEFINICIÓN DEL PROBLEMA
# ==============================================================================
# Queremos encontrar los pesos [w0, w1, w2, w3, w4, w5] tales que:
# (4*w0) + (-2*w1) + (3.5*w2) + (5*w3) + (-11*w4) + (-4.7*w5) = 44

function_inputs = [4.0, -2.0, 3.5, 5.0, -11.0, -4.7]
desired_output = 44.0


# ==============================================================================
# 2. FUNCIÓN DE APTITUD (FITNESS)
# ==============================================================================
def fitness_func(ga_instance, solution, solution_idx):
    # Calculamos la combinación lineal (producto escalar entre entradas y solución)
    output = np.sum(np.array(function_inputs) * solution)

    # Calculamos el error absoluto respecto al objetivo
    error = np.abs(output - desired_output)

    # PyGAD siempre MAXIMIZA el fitness.
    # Usamos la inversa del error: menor error -> mayor fitness.
    # Sumamos 1e-6 (épsilon) para evitar un error de división por cero si error == 0.
    fitness = 1.0 / (error + 1e-6)
    return fitness


# ==============================================================================
# 3. CONFIGURACIÓN DEL ALGORITMO GENÉTICO
# ==============================================================================
# Monitoreo del progreso cada 10 generaciones
def on_generation(ga_instance):
    gen = ga_instance.generations_completed
    if gen % 10 == 0:
        best_sol, best_fit, _ = ga_instance.best_solution()
        out = np.sum(np.array(function_inputs) * best_sol)
        print(f"Gen {gen:2d} | Mejor fitness: {best_fit:8.2f} | Salida actual: {out:.4f} (Objetivo: {desired_output})")


ga_instance = pygad.GA(
    # Ciclo de vida
    num_generations=50,          # Número de iteraciones evolutivas
    sol_per_pop=20,              # Tamaño de población (20 soluciones compitiendo)
    num_parents_mating=6,        # Padres seleccionados para reproducirse

    # Representación cromosómica
    num_genes=len(function_inputs),  # 6 genes (un peso por cada entrada)
    init_range_low=-4.0,             # Rango mínimo para la población inicial
    init_range_high=4.0,             # Rango máximo para la población inicial

    # Función de evaluación
    fitness_func=fitness_func,

    # Operadores genéticos
    parent_selection_type="sss",     # Steady State Selection (conserva los mejores)
    keep_parents=2,                  # Elitismo: número de padres que pasan intactos a la siguiente gen
    crossover_type="single_point",   # Cruce en un punto de corte
    mutation_type="random",          # Mutación aleatoria
    mutation_percent_genes=25,       # ~25% de los genes mutan (1-2 genes por hijo)

    # Callback de seguimiento
    on_generation=on_generation
)

# ==============================================================================
# 4. EJECUCIÓN
# ==============================================================================
print("Iniciando optimización...")
ga_instance.run()

# ==============================================================================
# 5. EXTRACCIÓN Y MUESTRA DE RESULTADOS
# ==============================================================================
solution, solution_fitness, _ = ga_instance.best_solution()
prediction = np.sum(np.array(function_inputs) * solution)
error_final = np.abs(prediction - desired_output)

print("\n" + "="*40)
print("RESULTADOS FINALES")
print("="*40)
print(f"Mejor vector de pesos (genes) : {np.round(solution, 4)}")
print(f"Valor de Fitness alcanzado    : {solution_fitness:.4f}")
print(f"Salida calculada              : {prediction:.6f}")
print(f"Valor deseado                 : {desired_output}")
print(f"Error absoluto                : {error_final:.6f}")