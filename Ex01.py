# EJEMPLO DE MINIMIZACIÓN PARÁBOLA SIMPLE

import pygad
import numpy as np

# DEFINCIÓN DE FUNCIONES

def fitness_dummy(ga_instance, solution, solution_idx):
    #Suma de los cuadrados (el mínimo global está en x_i=0)
    U = np.sum(solution**2)
    return -U 
    # PyGAD maximiza la salida para acercarla a 0
    
    
# Vamos a imprimir la mejor energía cada 50 generaciones    
def on_gen(ga_instance):
    if ga_instance.generations_completed % 50 == 0:
        best_fitness = ga_instance.best_solution()[1]
        print(f"Generacion {ga_instance.generations_completed}: Energia = {-best_fitness:.6f}")
        
ga = pygad.GA(num_generations=2000,
              num_parents_mating=10,
              fitness_func=fitness_dummy,
              sol_per_pop=50,
              num_genes=5,
              gene_space={'low':-10.0, 'high': 10.0},
              mutation_percent_genes=20,
              on_generation=on_gen)

# Comando de ejecución del algoritmo
ga.run()
# Mostrar la mejor solución al terminar
best_solution, best_fitness, _ = ga.best_solution()
print("\n--- Resultado Óptimo ---")
print(f"Vector x: {best_solution}")
print(f"Valor mínimo U(x): {-best_fitness:.6e}")