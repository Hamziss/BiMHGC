import pandas as pd
import numpy as np
import random
import math
from scipy.special import gamma
from typing import Optional
from .main import calculate_fitness, generate_random_bicluster, is_bicluster_valid

class SimulatedAnnealing:
    """
    Simulated Annealing optimization algorithm for bicluster discovery.

    This class implements a simulated annealing algorithm that iteratively improves
    a bicluster solution by accepting or rejecting neighboring solutions based on
    their fitness and the current temperature. The algorithm starts with high
    temperature allowing exploration of the solution space, then gradually cools
    down to converge on high-quality biclusters.

    Attributes:
        DGE_df (pd.DataFrame): Discretized gene expression dataframe
        min_row (int): Minimum number of rows (genes) required in valid biclusters
        min_col (int): Minimum number of columns (time points) required in valid biclusters
        initial_temperature (float): Starting temperature for annealing process
        final_temperature (float): Ending temperature (stopping criterion)
        cooling_rate (float): Rate at which temperature decreases each iteration
        max_iterations (int): Maximum number of iterations to perform
        neighborhood_size (int): Number of bits to potentially flip in each neighbor
        final_bicluster (list[tuple]): Tuple representing the best bicluster and its fitness score
                                        found after optimization, stored as (bicluster_array, fitness_score)
    """

    def __init__(self, SA_params: dict):
        """
        Initialize the Simulated Annealing optimizer.

        Args:
            DGE_df (pd.DataFrame): Discretized gene expression dataframe.
            min_row (int): Minimum number of genes required in valid biclusters.
            min_col (int): Minimum number of time points required in valid biclusters.
            initial_temperature (float): Starting temperature for annealing process.
            final_temperature (float): Ending temperature (stopping criterion).
            cooling_rate (float): Rate at which temperature decreases each iteration.
            max_iterations (int): Maximum number of iterations to perform.
            neighborhood_size (int): Number of bits to potentially flip in each neighbor.
        """
        self.DGE_df: pd.DataFrame = SA_params["DGE_df"]
        self.min_row: int = SA_params["min_row"]
        self.min_col: int = SA_params["min_col"]
        self.initial_temperature: float = SA_params["initial_temperature"]
        self.final_temperature: float = SA_params["final_temperature"]
        self.cooling_rate: float = SA_params["cooling_rate"]
        self.max_iterations: int = SA_params["max_iterations"]
        self.neighborhood_size: int = SA_params.get("neighborhood_size", 1)

        self.final_bicluster: tuple[np.ndarray, float]

    def _generate_neighbor(self, current_solution: np.ndarray) -> np.ndarray:
        """
        Generate a neighboring solution by flipping a small number of bits.

        Creates a neighbor by randomly selecting and flipping bits in the current
        solution. The number of bits flipped is determined by neighborhood_size.
        Ensures the resulting solution meets minimum bicluster constraints.

        Args:
            current_solution (np.ndarray): Current bicluster solution

        Returns:
            np.ndarray: Valid neighboring solution that satisfies minimum constraints
        """
        max_attempts = 100  # Prevent infinite loops
        attempts = 0

        while attempts < max_attempts:
            neighbor = current_solution.copy()

            # Randomly select bits to flip
            num_flips = random.randint(1, self.neighborhood_size)
            flip_indices = random.sample(range(len(neighbor)), num_flips)

            # Flip the selected bits
            for idx in flip_indices:
                neighbor[idx] = 1 - neighbor[idx]

            # Check if the neighbor is valid
            if is_bicluster_valid(neighbor, self.DGE_df, self.min_row, self.min_col):
                return neighbor

            attempts += 1

        # If we can't find a valid neighbor, return current solution
        return current_solution.copy()

    def _acceptance_probability(
        self, current_fitness: float, neighbor_fitness: float, temperature: float
    ) -> float:
        """
        Calculate the probability of accepting a neighbor solution.

        Uses the standard Metropolis criterion for simulated annealing.
        Always accepts better solutions (higher fitness), and accepts
        worse solutions with probability based on fitness difference and temperature.

        Args:
            current_fitness (float): Fitness of current solution
            neighbor_fitness (float): Fitness of neighbor solution
            temperature (float): Current temperature

        Returns:
            float: Acceptance probability between 0 and 1
        """
        if neighbor_fitness > current_fitness:
            return 1.0  # Always accept better solutions

        if temperature <= 0:
            return 0.0  # Avoid division by zero

        # Calculate acceptance probability for worse solutions
        fitness_diff = neighbor_fitness - current_fitness
        return math.exp(fitness_diff / temperature)

    def _update_temperature(self, current_temperature: float, iteration: int) -> float:
        """
        Update temperature using exponential cooling schedule.

        Implements exponential cooling: T(t) = T0 * (cooling_rate)^t
        where T0 is initial temperature and t is the iteration number.

        Args:
            current_temperature (float): Current temperature
            iteration (int): Current iteration number

        Returns:
            float: Updated temperature
        """
        # new_temperature = current_temperature * (self.cooling_rate**iteration)
        new_temperature = current_temperature * self.cooling_rate
        return new_temperature

    def optim(
        self, initial_bicluster: Optional[np.ndarray] = None, debug: bool = False
    ) -> None:
        """
        Execute the simulated annealing optimization to find optimal biclusters.

        Starts with an initial solution (random or provided) and iteratively
        improves it by exploring neighboring solutions. Acceptance of worse
        solutions decreases as temperature cools, allowing initial exploration
        followed by convergence to local optima.

        Args:
            initial_bicluster (Optional[np.ndarray]): Starting bicluster solution.
                                                     If None, generates random initial solution.
            debug (bool): If True, prints progress information during optimization.

        Side Effects:
            Updates self.final_bicluster and self.final_fitness with the best solution found

        Returns:
            None (results stored in self.final_bicluster and self.final_fitness)

        Algorithm Overview:
            1. Initialize solution (random or provided)
            2. Set initial temperature
            3. For each iteration:
               - Generate neighboring solution
               - Calculate acceptance probability
               - Accept or reject neighbor based on probability
               - Update temperature
               - Track best solution found
            4. Continue until temperature reaches final_temperature or max_iterations

        Convergence:
            - Stops when temperature drops below final_temperature
            - Or when max_iterations is reached
        """
        if debug:
            print("Launched Simulated Annealing optimization")

        # Initialize solution
        if initial_bicluster is None:
            current_solution = generate_random_bicluster(
                self.DGE_df, self.min_row, self.min_col
            )
            if debug:
                print("Generated random initial bicluster")
        else:
            current_solution = initial_bicluster.copy()
            if debug:
                print("Using provided initial bicluster")

        # Initialize best solution tracking
        current_fitness = calculate_fitness(current_solution, self.DGE_df)
        best_solution = current_solution.copy()
        best_fitness = current_fitness

        # Initialize temperature
        current_temperature = self.initial_temperature

        if debug:
            print(f"Initial fitness: {current_fitness:.6f}")
            print(f"Initial temperature: {current_temperature:.6f}")

        # Main optimization loop
        iteration = 0
        accepted_moves = 0
        temp_control = False

        while (
            current_temperature > self.final_temperature
            and iteration < self.max_iterations
        ):
            # Generate neighbor solution
            neighbor_solution = self._generate_neighbor(current_solution)
            neighbor_fitness = calculate_fitness(neighbor_solution, self.DGE_df)

            # Calculate acceptance probability
            accept_prob = self._acceptance_probability(
                current_fitness, neighbor_fitness, current_temperature
            )

            # Accept or reject the neighbor
            if random.random() < accept_prob:
                current_solution = neighbor_solution
                current_fitness = neighbor_fitness
                accepted_moves += 1
                temp_control = True

                # Update best solution if necessary
                if current_fitness > best_fitness:
                    best_solution = current_solution.copy()
                    best_fitness = current_fitness

            # Update temperature
            if temp_control:
                current_temperature = self._update_temperature(
                    current_temperature, iteration
                )
                temp_control = False

            # Print progress
            if debug and iteration % 1000 == 0:
                acceptance_rate = accepted_moves / (iteration + 1) * 100
                print(
                    f"Iteration {iteration}: Temperature = {current_temperature:.10f}, "
                    f"Current fitness = {current_fitness:.6f}, "
                    f"Best fitness = {best_fitness:.6f}, "
                    f"Acceptance rate = {acceptance_rate:.2f}%"
                )

            iteration += 1

        # Store final results
        self.final_bicluster = (best_solution, best_fitness)

        if debug:
            final_acceptance_rate = (
                accepted_moves / iteration * 100 if iteration > 0 else 0
            )
            print(f"\nOptimization completed after {iteration} iterations")
            print(f"Final temperature: {current_temperature:.10f}")
            print(f"Final best fitness: {best_fitness:.6f}")
            print(f"Total acceptance rate: {final_acceptance_rate:.2f}%")


class GeneticAlgorithm:
    """
    Genetic Algorithm optimization algorithm.

    This class implements a genetic algorithm that evolves a population of bicluster
    solutions to find high-quality biclusters.
    The algorithm uses tournament selection, uniform crossover, bit-flip mutation,
    and elitism to maintain population diversity while converging to optimal solutions.

    Attributes:
        DGE_df (pd.DataFrame): Discretized gene expression dataframe
        min_row (int): Minimum number of rows (genes) required in valid biclusters
        min_col (int): Minimum number of columns (time points) required in valid biclusters
        population_size (int): Number of individuals in the population
        result_size (int): Number of best solutions to return
        max_generations (int): Maximum number of generations to evolve
        crossover_rate (float): Probability of crossover between parents
        mutation_rate (float): Probability of mutating each bit
        elitism_ratio (float): Proportion of best individuals to preserve
        final_biclusters (list[tuple]): list of Top bicluster solutions after optimization
                                        stored as (bicluster_array, fitness_score)
    """

    def __init__(self, GA_params: dict):
        """
        Initialize the Genetic Algorithm optimizer.

        Args:
            DGE_df (pd.DataFrame): Discretized gene expression dataframe.
            min_row (int): Minimum number of genes required in valid biclusters.
            min_col (int): Minimum number of time points required in valid biclusters.
            population_size (int): Number of individuals in the population.
            result_size (int): Number of best solutions to return.
            max_generations (int): Maximum number of generations to evolve.
            crossover_rate (float): Probability of crossover between parents.
            mutation_rate (float): Probability of mutating each bit.
            elitism_ratio (float): Proportion of best individuals to preserve.
        """
        self.DGE_df: pd.DataFrame = GA_params["DGE_df"]
        self.min_row: int = GA_params["min_row"]
        self.min_col: int = GA_params["min_col"]
        self.population_size: int = GA_params["population_size"]
        self.result_size: int = GA_params["result_size"]
        self.max_generations: int = GA_params["max_generations"]
        self.crossover_rate: float = GA_params["crossover_rate"]
        self.mutation_rate: float = GA_params["mutation_rate"]
        self.elitism_ratio: float = GA_params["elitism_ratio"]

        self.final_biclusters: list[tuple]  # the best biclusters

    def _roulette_wheel_selection(
        self, population: list[np.ndarray], fitness_scores: list[float]
    ) -> np.ndarray:
        """
        Select a parent using roulette wheel selection.

        The probability of selecting an individual is proportional to its fitness
        value. Individuals with higher fitness have higher chances of being selected.

        Args:
            population (list[np.ndarray]): list of individuals (bicluster arrays)
            fitness_scores (list[float]): Corresponding fitness scores for each individual

        Returns:
            np.ndarray: Copy of the selected individual (bicluster solution)
        """

        # Calculate total fitness
        total_fitness = sum(fitness_scores)

        # If total fitness is zero (all individuals have zero fitness), use uniform selection
        if total_fitness <= 0:
            return random.choice(population).copy()

        # Generate a random point
        pick = random.uniform(0, total_fitness)

        # Spin the wheel
        current = 0
        for i, fitness in enumerate(fitness_scores):
            current += fitness
            if current >= pick:
                return population[i].copy()

    def _crossover(self, parent1: np.ndarray, parent2: np.ndarray):
        """
        Perform uniform crossover between two parents to produce two children.

        For each bit position, randomly decides whether to swap the bits between
        parents. This creates two offspring that combine features from both parents.
        Validates that children meet minimum bicluster constraints.

        Args:
            parent1 (np.ndarray): First parent bicluster solution
            parent2 (np.ndarray): Second parent bicluster solution

        Returns:
            tuple (tuple[np.ndarray, np.ndarray]): Two children as (child1, child2).If a child is
                                                   invalid, the corresponding parent is returned instead.
        """
        # Initialize children
        child1 = parent1.copy()
        child2 = parent2.copy()

        # Uniform crossover
        for i in range(len(parent1)):
            if random.random() < 0.5:
                child1[i] = parent2[i]
                child2[i] = parent1[i]

        # check if children are valid
        if not is_bicluster_valid(child1, self.DGE_df, self.min_row, self.min_col):
            child1 = parent1
        if not is_bicluster_valid(child2, self.DGE_df, self.min_row, self.min_col):
            child2 = parent2

        return child1, child2

    def _mutate(self, individual: np.ndarray, mutation_rate: float):
        """
        Mutate an individual by randomly flipping bits.

        Each bit in the individual has a probability equal to mutation_rate
        of being flipped (0→1 or 1→0). Ensures the mutated individual still
        meets minimum bicluster constraints.

        If mutation produces an invalid bicluster, the process repeats
        until a valid mutated individual is generated. This ensures
        all individuals in the population remain feasible.

        Args:
            individual (np.ndarray): Individual to mutate (bicluster array)
            mutation_rate (float): Probability of flipping each bit

        Returns:
            np.ndarray: Mutated individual that satisfies minimum constraints
        """
        while True:
            mutated = individual.copy()
            for i in range(len(mutated)):
                if random.random() < mutation_rate:
                    mutated[i] = 1 - mutated[i]  # Flip bit

            if is_bicluster_valid(mutated, self.DGE_df, self.min_row, self.min_col):
                break

        return mutated

    def optim(self, debug=False):
        """
        Execute the genetic algorithm optimization to find optimal biclusters.

        Evolves a population of bicluster solutions over multiple generations using
        selection, crossover, and mutation operations. Optionally incorporates
        hill climbing for local optimization of each individual.

        Args:
            use_hill_climbing (bool): If True, applies hill climbing to improve
                                      each individual in the population (default: False)
            debug (bool): If True, prints progress information during evolution (default: False)

        Side Effects:
            Updates self.final_biclusters with the top solutions found

        Returns:
            None (results stored in self.final_biclusters)

        Algorithm Overview:
            1. Initialize random population
            2. Optional: Apply hill climbing to initial population
            3. For each generation:
               - Evaluate fitness of all individuals
               - Select elite individuals for next generation
               - Generate offspring through selection, crossover, and mutation
               - Optional: Apply hill climbing to offspring
            4. Return top-ranking solutions

        Convergence:
            - Stops early if population converges (no improvement for 20 generations)
            - Otherwise runs for max_generations
        """
        if debug:
            print("Launched GA Optimization")

        # Initialize population
        population = [
            generate_random_bicluster(self.DGE_df, self.min_row, self.min_col)
            for _ in range(self.population_size)
        ]

        if debug:
            print(f"Initialized population of {self.population_size} individuals")

        # Number of elite individuals to keep
        num_elites = int(self.elitism_ratio * self.population_size)

        # Calculate fitness for each individual
        fitness_scores = [
            calculate_fitness(individual, self.DGE_df) for individual in population
        ]

        # Combine population with their fitness scores
        population_with_fitness = list(zip(population, fitness_scores))

        # Sort by fitness in descending order
        population_with_fitness.sort(key=lambda x: x[1], reverse=True)

        # Extract sorted population and scores
        sorted_population, sorted_fitness = zip(*population_with_fitness)

        # main loop
        for generation in range(self.max_generations):
            # Selection, crossover, and mutation to fill the rest of the population
            while len(population) < (self.population_size + num_elites):
                # Tournament selection
                parent1 = self._roulette_wheel_selection(
                    sorted_population, sorted_fitness
                )
                parent2 = self._roulette_wheel_selection(
                    sorted_population, sorted_fitness
                )

                # Crossover
                if random.random() < self.crossover_rate:
                    child1, child2 = self._crossover(parent1, parent2)
                else:
                    child1, child2 = parent1.copy(), parent2.copy()

                # Mutation
                child1 = self._mutate(child1, self.mutation_rate)
                child2 = self._mutate(child2, self.mutation_rate)

                population.append(child1)
                if len(population) < (self.population_size + num_elites):
                    population.append(child2)

            # Calculate fitness for each individual
            fitness_scores = [
                calculate_fitness(individual, self.DGE_df) for individual in population
            ]

            # Combine population with their fitness scores
            population_with_fitness = list(zip(population, fitness_scores))

            # Sort by fitness in descending order
            population_with_fitness.sort(key=lambda x: x[1], reverse=True)

            # Extract sorted population and scores
            sorted_population, sorted_fitness = zip(*population_with_fitness)

            # Elitism: keep the best individuals
            population = list(sorted_population[:num_elites])

            # Print progress every 10 generations
            if debug and generation % 10 == 0:
                print(f"Generation {generation}: Best fitness = {sorted_fitness[0]}")

            # Check for convergence
            if generation > 20:
                # If best fitness hasn't improved significantly in the last 20 generations
                if sorted_fitness[0] == sorted_fitness[-1]:
                    print(f"Converged at generation {generation+1}")
                    break

        # Return top solutions
        self.final_biclusters = population_with_fitness[: self.result_size]


class CuckooSearch:
    """
    Cuckoo Search optimization algorithm for bicluster discovery.

    This class implements the Cuckoo Search algorithm that evolves a population of bicluster
    solutions to find high-quality biclusters. The algorithm mimics the brood parasitism
    behavior of cuckoo birds, using Lévy flights for exploration and abandonment of worst
    nests to maintain population diversity while converging to optimal solutions.

    Attributes:
        DGE_df (pd.DataFrame): Discretized gene expression dataframe
        min_row (int): Minimum number of rows (genes) required in valid biclusters
        min_col (int): Minimum number of columns (time points) required in valid biclusters
        population_size (int): Number of nests (solutions) in the population
        result_size (int): Number of best solutions to return
        max_generations (int): Maximum number of generations to evolve
        discovery_rate (float): Probability of discovering alien eggs (abandonment rate)
        levy_alpha (float): Lévy flight parameter (stability parameter)
        levy_beta (float): Lévy flight parameter (scale parameter)
        final_biclusters (list[tuple]): list of top bicluster solutions after optimization
                                        stored as (bicluster_array, fitness_score)
    """

    def __init__(self, CS_params: dict):
        """
        Initialize the Cuckoo Search optimizer.

        Args:
            CS_params (dict): Dictionary containing algorithm parameters:
                - DGE_df (pd.DataFrame): Discretized gene expression dataframe
                - min_row (int): Minimum number of genes required in valid biclusters
                - min_col (int): Minimum number of time points required in valid biclusters
                - population_size (int): Number of nests in the population
                - result_size (int): Number of best solutions to return
                - max_generations (int): Maximum number of generations to evolve
                - discovery_rate (float): Probability of discovering alien eggs (default: 0.25)
                - levy_alpha (float): Lévy flight stability parameter (default: 1.5)
                - levy_beta (float): Lévy flight scale parameter (default: 1.0)
        """
        self.DGE_df: pd.DataFrame = CS_params["DGE_df"]
        self.min_row: int = CS_params["min_row"]
        self.min_col: int = CS_params["min_col"]
        self.population_size: int = CS_params["population_size"]
        self.result_size: int = CS_params.get("result_size", 30)
        self.max_generations: int = CS_params["max_generations"]
        self.discovery_rate: float = CS_params.get("discovery_rate", 0.25)
        self.levy_alpha: float = CS_params.get("levy_alpha", 1.5)
        self.levy_beta: float = CS_params.get("levy_beta", 1.0)

        self.final_biclusters: list[tuple]  # the best biclusters

    def _levy_flight(self, dimension: int) -> np.ndarray:
        """
        Generate Lévy flight step for cuckoo movement.

        Lévy flights are random walks characterized by heavy-tailed probability
        distributions, providing an efficient search strategy that balances
        local and global exploration.

        Args:
            dimension (int): Dimensionality of the step vector

        Returns:
            np.ndarray: Lévy flight step vector
        """
        # Calculate sigma for Lévy distribution
        num = gamma(1 + self.levy_alpha) * np.sin(np.pi * self.levy_alpha / 2)
        den = (
            gamma((1 + self.levy_alpha) / 2)
            * self.levy_alpha
            * (2 ** ((self.levy_alpha - 1) / 2))
        )
        sigma = (num / den) ** (1 / self.levy_alpha)

        # Generate Lévy flight
        u = np.random.normal(0, sigma, dimension)
        v = np.random.normal(0, 1, dimension)
        step = u / (np.abs(v) ** (1 / self.levy_alpha))

        return step * self.levy_beta

    def _generate_new_solution_levy(
        self, current_nest: np.ndarray, best_nest: np.ndarray
    ) -> np.ndarray:
        """
        Generate a new solution using Lévy flight around the current nest.

        The new solution is generated by taking a Lévy flight step from the current
        nest towards the best nest, with some random exploration.

        Args:
            current_nest (np.ndarray): Current nest (bicluster solution)
            best_nest (np.ndarray): Best nest found so far

        Returns:
            np.ndarray: New bicluster solution that satisfies minimum constraints
        """
        max_attempts = 50
        attempts = 0

        while attempts < max_attempts:
            # Generate Lévy flight step
            levy_step = self._levy_flight(len(current_nest))

            # Scale the step size
            step_size = 0.01 * levy_step

            # Move towards best nest with Lévy flight perturbation
            direction = best_nest.astype(float) - current_nest.astype(float)
            new_solution_float = (
                current_nest.astype(float) + step_size + 0.1 * direction
            )

            # Convert to binary using probabilistic approach
            probabilities = 1 / (1 + np.exp(-new_solution_float))  # Sigmoid function
            new_solution = (np.random.random(len(current_nest)) < probabilities).astype(
                int
            )

            # Check if solution meets minimum requirements
            if is_bicluster_valid(
                new_solution, self.DGE_df, self.min_row, self.min_col
            ):
                return new_solution

            attempts += 1

        # If no valid solution found, return a random valid bicluster
        return generate_random_bicluster(self.DGE_df, self.min_row, self.min_col)

    def _abandon_worst_nests(
        self, population: list[np.ndarray], fitness_scores: list[float]
    ) -> list[np.ndarray]:
        """
        Abandon worst nests and replace them with new random solutions.

        This operation simulates the discovery of alien eggs by host birds,
        leading to abandonment of nests and construction of new ones.

        Args:
            population (list[np.ndarray]): Current population of nests
            fitness_scores (list[float]): Corresponding fitness scores

        Returns:
            list[np.ndarray]: Updated population with worst nests replaced
        """
        # Sort population by fitness (ascending order to identify worst)
        population_with_fitness = list(zip(population, fitness_scores))
        population_with_fitness.sort(key=lambda x: x[1])

        # Calculate number of nests to abandon
        num_abandon = int(self.discovery_rate * self.population_size)

        new_population = []
        new_fitness_scores = []

        for i, (nest, fitness) in enumerate(population_with_fitness):
            if i < num_abandon:  # Abandon worst nests
                # Generate new random nest
                new_nest = generate_random_bicluster(
                    self.DGE_df, self.min_row, self.min_col
                )
                new_population.append(new_nest)
                new_fitness_scores.append(calculate_fitness(new_nest, self.DGE_df))
            else:
                new_population.append(nest)
                new_fitness_scores.append(fitness)

        return new_population, new_fitness_scores

    def optim(self, debug=False):
        """
        Execute the Cuckoo Search optimization to find optimal biclusters.

        Evolves a population of bicluster solutions (nests) over multiple generations using
        Lévy flights for exploration and abandonment of worst solutions. Optionally
        incorporates hill climbing for local optimization.

        Args:
            use_hill_climbing (bool): If True, applies hill climbing to improve
                                      solutions (default: False)
            debug (bool): If True, prints progress information during evolution (default: False)

        Side Effects:
            Updates self.final_biclusters with the top solutions found

        Returns:
            None (results stored in self.final_biclusters)

        Algorithm Overview:
            1. Initialize random population of nests
            2. Optional: Apply hill climbing to initial population
            3. For each generation:
               - Generate new solutions via Lévy flights
               - Replace solutions if fitness improves
               - Abandon worst nests with probability discovery_rate
               - Optional: Apply hill climbing to new solutions
            4. Return top-ranking solutions

        Convergence:
            - Stops early if population converges (no improvement for 30 generations)
            - Otherwise runs for max_generations
        """
        if debug:
            print("Launched Cuckoo Search optimization")

        # Initialize population of nests
        population = [
            generate_random_bicluster(self.DGE_df, self.min_row, self.min_col)
            for _ in range(self.population_size)
        ]

        if debug:
            print(f"Initialized population of {self.population_size} nests")

        # Calculate fitness for each nest
        fitness_scores = [calculate_fitness(nest, self.DGE_df) for nest in population]

        # Find best nest
        best_index = np.argmax(fitness_scores)
        best_nest = population[best_index]
        best_fitness = fitness_scores[best_index]

        # Main loop
        for iteration in range(self.max_generations):
            # Generate new solutions via Lévy flights
            new_population = []
            new_fitness_scores = []
            for i, current_nest in enumerate(population):
                # Generate new solution using Lévy flight
                new_solution = self._generate_new_solution_levy(current_nest, best_nest)

                # Calculate fitness of new solution
                new_fitness = calculate_fitness(new_solution, self.DGE_df)

                # Replace if better
                if new_fitness > fitness_scores[i]:
                    new_population.append(new_solution)
                    new_fitness_scores.append(new_fitness)
                else:
                    new_population.append(current_nest)
                    new_fitness_scores.append(fitness_scores[i])

            # Update population
            population = new_population
            fitness_scores = new_fitness_scores

            # Abandon worst nests and replace with new random solutions
            population, fitness_scores = self._abandon_worst_nests(
                population, fitness_scores
            )

            # Find best nest
            best_index = np.argmax(fitness_scores)
            best_nest = population[best_index]
            best_fitness = fitness_scores[best_index]

            # Print progress
            if debug and iteration % 10 == 0:
                print(
                    f"Iteration {iteration}: Best fitness = {(best_fitness):.6f}, "
                    f"Average fitness = {(sum(fitness_scores) / len(fitness_scores)):.6f}"
                )

        # Combine population with fitness scores
        final_population_with_fitness = list(zip(population, fitness_scores))

        # Sort by fitness in descending order (maximize)
        final_population_with_fitness.sort(key=lambda x: x[1], reverse=True)

        # Return top solutions
        self.final_biclusters = final_population_with_fitness[: self.result_size]

        if debug:
            print("CS DONE")


class GeneticAlgorithmSA:
    """
    Genetic Algorithm optimization algorithm.

    This class implements a genetic algorithm that evolves a population of bicluster
    solutions to find high-quality biclusters.
    The algorithm uses tournament selection, uniform crossover, bit-flip mutation,
    and elitism to maintain population diversity while converging to optimal solutions.

    Attributes:
        DGE_df (pd.DataFrame): Discretized gene expression dataframe
        min_row (int): Minimum number of rows (genes) required in valid biclusters
        min_col (int): Minimum number of columns (time points) required in valid biclusters
        population_size (int): Number of individuals in the population
        result_size (int): Number of best solutions to return
        max_generations (int): Maximum number of generations to evolve
        crossover_rate (float): Probability of crossover between parents
        mutation_rate (float): Probability of mutating each bit
        elitism_ratio (float): Proportion of best individuals to preserve
        final_biclusters (list[tuple]): list of Top bicluster solutions after optimization
                                        stored as (bicluster_array, fitness_score)
    """

    def __init__(self, GA_params: dict, SA_params: dict):
        """
        Initialize the Genetic Algorithm optimizer.

        Args:
            DGE_df (pd.DataFrame): Discretized gene expression dataframe.
            min_row (int): Minimum number of genes required in valid biclusters.
            min_col (int): Minimum number of time points required in valid biclusters.
            population_size (int): Number of individuals in the population.
            result_size (int): Number of best solutions to return.
            max_generations (int): Maximum number of generations to evolve.
            crossover_rate (float): Probability of crossover between parents.
            mutation_rate (float): Probability of mutating each bit.
            elitism_ratio (float): Proportion of best individuals to preserve.
        """
        self.DGE_df: pd.DataFrame = GA_params["DGE_df"]
        self.min_row: int = GA_params["min_row"]
        self.min_col: int = GA_params["min_col"]
        self.population_size: int = GA_params["population_size"]
        self.result_size: int = GA_params["result_size"]
        self.max_generations: int = GA_params["max_generations"]
        self.crossover_rate: float = GA_params["crossover_rate"]
        self.mutation_rate: float = GA_params["mutation_rate"]
        self.elitism_ratio: float = GA_params["elitism_ratio"]
        self.SA_params: dict = SA_params

        self.final_biclusters: list[tuple]  # the best biclusters

    def _roulette_wheel_selection(
        self, population: list[np.ndarray], fitness_scores: list[float]
    ) -> np.ndarray:
        """
        Select a parent using roulette wheel selection.

        The probability of selecting an individual is proportional to its fitness
        value. Individuals with higher fitness have higher chances of being selected.

        Args:
            population (list[np.ndarray]): list of individuals (bicluster arrays)
            fitness_scores (list[float]): Corresponding fitness scores for each individual

        Returns:
            np.ndarray: Copy of the selected individual (bicluster solution)
        """

        # Calculate total fitness
        total_fitness = sum(fitness_scores)

        # If total fitness is zero (all individuals have zero fitness), use uniform selection
        if total_fitness <= 0:
            return random.choice(population).copy()

        # Generate a random point
        pick = random.uniform(0, total_fitness)

        # Spin the wheel
        current = 0
        for i, fitness in enumerate(fitness_scores):
            current += fitness
            if current >= pick:
                return population[i].copy()

    def _crossover(self, parent1: np.ndarray, parent2: np.ndarray):
        """
        Perform uniform crossover between two parents to produce two children.

        For each bit position, randomly decides whether to swap the bits between
        parents. This creates two offspring that combine features from both parents.
        Validates that children meet minimum bicluster constraints.

        Args:
            parent1 (np.ndarray): First parent bicluster solution
            parent2 (np.ndarray): Second parent bicluster solution

        Returns:
            tuple (tuple[np.ndarray, np.ndarray]): Two children as (child1, child2).If a child is
                                                   invalid, the corresponding parent is returned instead.
        """
        # Initialize children
        child1 = parent1.copy()
        child2 = parent2.copy()

        # Uniform crossover
        for i in range(len(parent1)):
            if random.random() < 0.5:
                child1[i] = parent2[i]
                child2[i] = parent1[i]

        # check if children are valid
        if not is_bicluster_valid(child1, self.DGE_df, self.min_row, self.min_col):
            child1 = parent1
        if not is_bicluster_valid(child2, self.DGE_df, self.min_row, self.min_col):
            child2 = parent2

        return child1, child2

    def _mutate(self, individual: np.ndarray, mutation_rate: float):
        """
        Mutate an individual by randomly flipping bits.

        Each bit in the individual has a probability equal to mutation_rate
        of being flipped (0→1 or 1→0). Ensures the mutated individual still
        meets minimum bicluster constraints.

        If mutation produces an invalid bicluster, the process repeats
        until a valid mutated individual is generated. This ensures
        all individuals in the population remain feasible.

        Args:
            individual (np.ndarray): Individual to mutate (bicluster array)
            mutation_rate (float): Probability of flipping each bit

        Returns:
            np.ndarray: Mutated individual that satisfies minimum constraints
        """
        while True:
            mutated = individual.copy()
            for i in range(len(mutated)):
                if random.random() < mutation_rate:
                    mutated[i] = 1 - mutated[i]  # Flip bit

            if is_bicluster_valid(mutated, self.DGE_df, self.min_row, self.min_col):
                break

        return mutated

    def _remove_worst_individuals(
        self, population: list[np.ndarray], fitness_scores: list[float]
    ):
        """
        Remove the two worst individuals from the population.

        Args:
            population (list[np.ndarray]): List of individuals (bicluster arrays)
            fitness_scores (list[float]): Corresponding fitness scores for each individual

        Returns:
            tuple: (updated_population, updated_fitness_scores) with two worst individuals removed
        """
        # Combine population with fitness scores
        pop_with_fitness = list(zip(population, fitness_scores))

        # Sort by fitness (ascending - worst first)
        pop_with_fitness.sort(key=lambda x: x[1])

        # Remove two worst individuals
        pop_with_fitness = pop_with_fitness[2:]

        # Separate population and fitness scores
        updated_population, updated_fitness = (
            zip(*pop_with_fitness) if pop_with_fitness else ([], [])
        )

        return list(updated_population), list(updated_fitness)

    def optim(self, debug=False):
        """
        Execute the memetic algorithm optimization to find optimal biclusters.

        Implements Algorithm 1: Memetic algorithm combining genetic operations
        with simulated annealing local search. Maintains a fixed population size
        of 400 individuals and uses local optimization on offspring.

        Args:
            debug (bool): If True, prints progress information during evolution (default: False)

        Side Effects:
            Updates self.final_biclusters with the top solutions found

        Returns:
            None (results stored in self.final_biclusters)

        Algorithm Overview:
            1. Initialize random population of 400 individuals
            2. For each generation:
               - Select two parents using tournament selection
               - Create two children through crossover
               - Mutate children
               - Evaluate children fitness
               - Apply simulated annealing local search to optimize children
               - Insert optimized children into population
               - Remove two worst individuals from population
            3. Sort population by fitness and return top solutions

        Convergence:
            - Runs for max_generations
            - Early stopping if fitness improvement plateaus
        """
        if debug:
            print("Launched Memetic Algorithm optimization")

        # Line 4: Initialize population with 400 random solutions
        population = [
            generate_random_bicluster(self.DGE_df, self.min_row, self.min_col)
            for _ in range(self.population_size)
        ]

        if debug:
            print(f"Initialized population of {self.population_size} individuals")

        # Calculate initial fitness scores
        fitness_scores = [
            calculate_fitness(individual, self.DGE_df) for individual in population
        ]

        sa = SimulatedAnnealing(self.SA_params)

        # Lines 6-15: Main loop of memetic algorithm
        for generation in range(self.max_generations):
            # Line 7: Select two individuals as parents
            parent1 = self._roulette_wheel_selection(population, fitness_scores)
            parent2 = self._roulette_wheel_selection(population, fitness_scores)

            # Line 8: Create two children by applying crossover operator
            if random.random() < self.crossover_rate:
                child1, child2 = self._crossover(parent1, parent2)
            else:
                child1, child2 = parent1.copy(), parent2.copy()

            # Line 9: Mutate children
            child1 = self._mutate(child1, self.mutation_rate)
            child2 = self._mutate(child2, self.mutation_rate)

            # Line 11: Local search to optimize children using Simulated Annealing
            # Optimize child1
            sa.optim(initial_bicluster=child1, debug=False)
            optimized_child1, optimized_child1_fitness = sa.final_bicluster

            # Optimize child2
            sa.optim(initial_bicluster=child2, debug=False)
            optimized_child2, optimized_child2_fitness = sa.final_bicluster

            # Line 12: Insert optimized children into population
            population.extend([optimized_child1, optimized_child2])
            fitness_scores.extend([optimized_child1_fitness, optimized_child2_fitness])

            # Line 13: Remove two worst individuals from population
            population, fitness_scores = self._remove_worst_individuals(
                population, fitness_scores
            )

            # Print progress every 10 generations
            if debug and generation % 10 == 0:
                print(
                    f"Generation {generation}: Best fitness = {(max(fitness_scores)):.6f}, "
                    f"Average fitness = {(sum(fitness_scores) / len(fitness_scores)):.6f}"
                )

        del sa

        # Line 16: Sort population by fitness values
        final_population_with_fitness = list(zip(population, fitness_scores))
        final_population_with_fitness.sort(key=lambda x: x[1], reverse=True)

        # Line 17: Return top solutions as final GE biclusters
        self.final_biclusters = final_population_with_fitness[: self.result_size]

        if debug:
            print(f"Optimization completed after {generation} generations")
            print(f"Best fitness achieved: {self.final_biclusters[0][1]:.6f}")


class CuckooSearchSA:
    """
    Cuckoo Search optimization algorithm for bicluster discovery.

    This class implements the Cuckoo Search algorithm that evolves a population of bicluster
    solutions to find high-quality biclusters. The algorithm mimics the brood parasitism
    behavior of cuckoo birds, using Lévy flights for exploration and abandonment of worst
    nests to maintain population diversity while converging to optimal solutions.

    Attributes:
        DGE_df (pd.DataFrame): Discretized gene expression dataframe
        min_row (int): Minimum number of rows (genes) required in valid biclusters
        min_col (int): Minimum number of columns (time points) required in valid biclusters
        population_size (int): Number of nests (solutions) in the population
        result_size (int): Number of best solutions to return
        max_generations (int): Maximum number of generations to evolve
        discovery_rate (float): Probability of discovering alien eggs (abandonment rate)
        levy_alpha (float): Lévy flight parameter (stability parameter)
        levy_beta (float): Lévy flight parameter (scale parameter)
        final_biclusters (list[tuple]): list of top bicluster solutions after optimization
                                        stored as (bicluster_array, fitness_score)
    """

    def __init__(self, CS_params: dict, SA_params: dict):
        """
        Initialize the Cuckoo Search optimizer.

        Args:
            CS_params (dict): Dictionary containing algorithm parameters:
                - DGE_df (pd.DataFrame): Discretized gene expression dataframe
                - min_row (int): Minimum number of genes required in valid biclusters
                - min_col (int): Minimum number of time points required in valid biclusters
                - population_size (int): Number of nests in the population
                - result_size (int): Number of best solutions to return
                - max_generations (int): Maximum number of generations to evolve
                - discovery_rate (float): Probability of discovering alien eggs (default: 0.25)
                - levy_alpha (float): Lévy flight stability parameter (default: 1.5)
                - levy_beta (float): Lévy flight scale parameter (default: 1.0)
        """
        self.DGE_df: pd.DataFrame = CS_params["DGE_df"]
        self.min_row: int = CS_params["min_row"]
        self.min_col: int = CS_params["min_col"]
        self.population_size: int = CS_params["population_size"]
        self.result_size: int = CS_params.get("result_size", 30)
        self.max_generations: int = CS_params["max_generations"]
        self.discovery_rate: float = CS_params.get("discovery_rate", 0.25)
        self.levy_alpha: float = CS_params.get("levy_alpha", 1.5)
        self.levy_beta: float = CS_params.get("levy_beta", 1.0)
        self.SA_params = SA_params

        self.final_biclusters: list[tuple]  # the best biclusters

    def _levy_flight(self, dimension: int) -> np.ndarray:
        """
        Generate Lévy flight step for cuckoo movement.

        Lévy flights are random walks characterized by heavy-tailed probability
        distributions, providing an efficient search strategy that balances
        local and global exploration.

        Args:
            dimension (int): Dimensionality of the step vector

        Returns:
            np.ndarray: Lévy flight step vector
        """
        # Calculate sigma for Lévy distribution
        num = gamma(1 + self.levy_alpha) * np.sin(np.pi * self.levy_alpha / 2)
        den = (
            gamma((1 + self.levy_alpha) / 2)
            * self.levy_alpha
            * (2 ** ((self.levy_alpha - 1) / 2))
        )
        sigma = (num / den) ** (1 / self.levy_alpha)

        # Generate Lévy flight
        u = np.random.normal(0, sigma, dimension)
        v = np.random.normal(0, 1, dimension)
        step = u / (np.abs(v) ** (1 / self.levy_alpha))

        return step * self.levy_beta

    def _get_random_nest_index(self, exclude_index: int = -1) -> int:
        """
        Get a random nest index, optionally excluding a specific index.

        Args:
            exclude_index (int): Index to exclude from selection (default: -1, no exclusion)

        Returns:
            int: Random nest index
        """
        available_indices = [
            i for i in range(self.population_size) if i != exclude_index
        ]
        return random.choice(available_indices)

    def _generate_new_solution_levy(
        self, current_nest: np.ndarray, best_nest: np.ndarray
    ) -> np.ndarray:
        """
        Generate a new solution using Lévy flight around the current nest.

        The new solution is generated by taking a Lévy flight step from the current
        nest towards the best nest, with some random exploration.

        Args:
            current_nest (np.ndarray): Current nest (bicluster solution)
            best_nest (np.ndarray): Best nest found so far

        Returns:
            np.ndarray: New bicluster solution that satisfies minimum constraints
        """
        max_attempts = 50
        attempts = 0

        while attempts < max_attempts:
            # Generate Lévy flight step
            levy_step = self._levy_flight(len(current_nest))

            # Scale the step size
            step_size = 0.01 * levy_step

            # Move towards best nest with Lévy flight perturbation
            direction = best_nest.astype(float) - current_nest.astype(float)
            new_solution_float = (
                current_nest.astype(float) + step_size + 0.1 * direction
            )

            # Convert to binary using probabilistic approach
            probabilities = 1 / (1 + np.exp(-new_solution_float))  # Sigmoid function
            new_solution = (np.random.random(len(current_nest)) < probabilities).astype(
                int
            )

            # Check if solution meets minimum requirements
            if is_bicluster_valid(
                new_solution, self.DGE_df, self.min_row, self.min_col
            ):
                return new_solution

            attempts += 1

        # If no valid solution found, return a random valid bicluster
        return generate_random_bicluster(self.DGE_df, self.min_row, self.min_col)

    def _abandon_worst_nests(
        self, population: list[np.ndarray], fitness_scores: list[float]
    ) -> list[np.ndarray]:
        """
        Abandon worst nests and replace them with new random solutions.

        This operation simulates the discovery of alien eggs by host birds,
        leading to abandonment of nests and construction of new ones.

        Args:
            population (list[np.ndarray]): Current population of nests
            fitness_scores (list[float]): Corresponding fitness scores

        Returns:
            list[np.ndarray]: Updated population with worst nests replaced
        """
        # Sort population by fitness (ascending order to identify worst)
        population_with_fitness = list(zip(population, fitness_scores))
        population_with_fitness.sort(key=lambda x: x[1])

        # Calculate number of nests to abandon
        num_abandon = int(self.discovery_rate * self.population_size)

        new_population = []
        new_fitness_scores = []

        for i, (nest, fitness) in enumerate(population_with_fitness):
            if i < num_abandon:  # Abandon worst nests
                # Generate new random nest
                new_nest = generate_random_bicluster(
                    self.DGE_df, self.min_row, self.min_col
                )
                new_population.append(new_nest)
                new_fitness_scores.append(calculate_fitness(new_nest, self.DGE_df))
            else:
                new_population.append(nest)
                new_fitness_scores.append(fitness)

        return new_population, new_fitness_scores

    def optim(self, debug=False):
        """
        Execute the Cuckoo Search optimization to find optimal biclusters.

        Evolves a population of bicluster solutions (nests) over multiple generations using
        Lévy flights for exploration and abandonment of worst solutions. Optionally
        incorporates hill climbing for local optimization.

        Args:
            use_hill_climbing (bool): If True, applies hill climbing to improve
                                      solutions (default: False)
            debug (bool): If True, prints progress information during evolution (default: False)

        Side Effects:
            Updates self.final_biclusters with the top solutions found

        Returns:
            None (results stored in self.final_biclusters)

        Algorithm Overview:
            1. Initialize random population of nests
            2. Optional: Apply hill climbing to initial population
            3. For each generation:
               - Generate new solutions via Lévy flights
               - Replace solutions if fitness improves
               - Abandon worst nests with probability discovery_rate
               - Optional: Apply hill climbing to new solutions
            4. Return top-ranking solutions

        Convergence:
            - Stops early if population converges (no improvement for 30 generations)
            - Otherwise runs for max_generations
        """
        if debug:
            print("Launched Cuckoo Search optimization")

        # Initialize population of nests
        population = [
            generate_random_bicluster(self.DGE_df, self.min_row, self.min_col)
            for _ in range(self.population_size)
        ]

        if debug:
            print(f"Initialized population of {self.population_size} nests")

        # Calculate fitness for each nest
        fitness_scores = [calculate_fitness(nest, self.DGE_df) for nest in population]

        # Find best nest
        best_index = np.argmax(fitness_scores)
        best_nest = population[best_index]
        best_fitness = fitness_scores[best_index]

        sa = SimulatedAnnealing(self.SA_params)

        # Main optimization loop
        for iteration in range(self.max_generations):
            # choose random nest from population
            random_index = self._get_random_nest_index(best_index)
            random_nest = population[random_index]

            # Generate new solution using Lévy flight
            new_solution = self._generate_new_solution_levy(random_nest, best_nest)

            # improve using SA
            sa.optim(new_solution)
            new_solution, new_solution_fitness = sa.final_bicluster

            # Replace if better
            if new_solution_fitness > fitness_scores[random_index]:
                population[random_index] = new_solution
                fitness_scores[random_index] = new_solution_fitness

            # Abandon worst nests and replace with new random solutions
            population, fitness_scores = self._abandon_worst_nests(
                population, fitness_scores
            )

            # Find best nest
            best_index = np.argmax(fitness_scores)
            best_nest = population[best_index]
            best_fitness = fitness_scores[best_index]

            # Print progress
            if debug and iteration % 10 == 0:
                print(
                    f"Iteration {iteration}: Best fitness = {(best_fitness):.6f}, "
                    f"Average fitness = {(sum(fitness_scores) / len(fitness_scores)):.6f}"
                )

        # Combine population with fitness scores
        final_population_with_fitness = list(zip(population, fitness_scores))

        # Sort by fitness in descending order
        final_population_with_fitness.sort(key=lambda x: x[1], reverse=True)

        # Return top solutions
        self.final_biclusters = final_population_with_fitness[: self.result_size]
