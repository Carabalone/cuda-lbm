# Creating New Scenarios:

Lets add a new simulation scenario (let's call it `MyNewScenario`):

1.  **Create Scenario Directory:**
    * Navigate to the `src/scenarios/` directory.
    * Create a new subdirectory named `myNewScenario`.

2.  **Define Scenario Functors (`src/scenarios/myNewScenario/myNewScenarioFunctors.cuh`):**
    * Create a new header file named `myNewScenarioFunctors.cuh` inside the `src/scenarios/myNewScenario/` directory.
    * In this file, define the necessary functors:
        * `MyNewScenarioInit`: This struct will define the initial conditions (e.g., initial density, velocity) for your scenario. It should have an `operator()` that takes `float* rho, float* u, float* force, int node` and sets the initial values. Remember to include the `apply_forces(float* rho, float* u, float* force, int node)` function, even if it just zeroes out forces, especially if you might use IBM features.
        * `MyNewScenarioBoundary`: This struct defines the boundary conditions. It should have an `operator()` that takes `int x, int y` and returns a boundary flag (e.g., `BC_flag::BOUNCE_BACK`, `BC_flag::FLUID`, custom flags) indicating the type of boundary at that location.
        * `MyNewScenarioValidation` (Optional): If your scenario has an analytical solution for comparison, define this struct to compute the error. If not, you can omit it or use a placeholder like `void`.

3.  **Define Scenario Traits (`src/scenarios/myNewScenario/myNewScenarioScenario.cuh`):**
    * Create another header file named `myNewScenarioScenario.cuh` inside `src/scenarios/myNewScenario/`.
    * Define a struct `MyNewScenarioScenario` that inherits from `ScenarioTrait`.
    * Specify the template arguments for `ScenarioTrait` using the functors you just defined:
        ```cpp
        #include "scenarios/scenario.cuh"
        #include "myNewScenarioFunctors.cuh"
        #include "functors/includes.cuh" // Include necessary functor implementations

        struct MyNewScenarioScenario : public ScenarioTrait<
            MyNewScenarioInit,         // Your Init functor
            MyNewScenarioBoundary,     // Your Boundary functor
            MyNewScenarioValidation,   // Your Validation functor (or void)
            BGK                        // Choose Collision Operator (e.g., BGK, MRT, CM) 
                                       // For the CM operator, you need an adapter, such as CM<NoAdapter> or
                                       // CM<OptimalAdapter>
        > {
            // Override default parameters from ScenarioTrait as needed
            static constexpr float viscosity = /* your value */; // e.g., 0.01f 
            static constexpr float tau = viscosity_to_tau(viscosity); 
            static constexpr float omega = 1.0f / tau; 
            static constexpr float u_max = /* your characteristic velocity */; // e.g., 0.1f 

            // If using MRT/CM, define the relaxation parameters S matrix 
            // follow the other scenarios for examples.
            // static constexpr float S[quadratures] = { /* your relaxation rates */ };

            // Define a name for the scenario
            static const char* name() { return "MyNewScenario"; }  

            // Provide static methods to get instances of your functors
            __host__ __device__ static InitType init() { return InitType(/* constructor args if any */); } 
            static BoundaryType boundary() { return BoundaryType(/* constructor args if any */); }  
            static ValidationType validation() { return ValidationType(/* constructor args if any */); } 

            // If using IBM, implement add_bodies to add IBMBody objects 
            // static void add_bodies() {
            //     IBM_bodies.push_back(create_shape(...));
            // }

            // Implement compute_error if using validation 
            // template <typename LBMSolver>
            // static float compute_error(LBMSolver& solver) {
            //     // ... implementation ...
            //     return error;
            // }
        };
        ```

4.  **Update `defines.hpp`:**
    * Open `src/defines.hpp`.
    * Add a new define for your scenario: `#define USE_MYNEWSCENARIO`.
    * **Comment out** the defines for other scenarios (like `USE_POISEUILLE`, `USE_LID_DRIVEN`) to ensure only your new scenario is active.
    * Add an `#elif defined(USE_MYNEWSCENARIO)` block to define simulation parameters like `NX` and `NY` if they differ from defaults.
        ```cpp
        // #define USE_TAYLOR_GREEN
        // #define USE_POISEUILLE
        // #define USE_LID_DRIVEN
        #define USE_MYNEWSCENARIO // Activate your new scenario

        // simulation parameters
        #ifdef USE_TAYLOR_GREEN
            // ...
        #elif defined(USE_POISEUILLE)
            // ...
        #elif defined(USE_LID_DRIVEN)
            // ...
        #elif defined(USE_MYNEWSCENARIO) // Add section for your scenario
            #define SCALE 1
            #define NX (/* Your X dimension */ * SCALE) // e.g., 200
            #define NY (/* Your Y dimension */ * SCALE) // e.g., 100
        #endif
        ```

5.  **Update `main.cu`:**
    * Open `src/main.cu`.
    * Add an `#elif` block to include your scenario header and define the `Scenario` type alias when `USE_MYNEWSCENARIO` is defined.
        ```cpp
        #if defined(USE_TAYLOR_GREEN)
            #include "scenarios/taylorGreen/TaylorGreenScenario.cuh"
            using Scenario = TaylorGreenScenario;
        #elif defined(USE_POISEUILLE)
            #include "scenarios/poiseuille/PoiseuilleScenario.cuh"
            using Scenario = PoiseuilleScenario;
        #elif defined(USE_LID_DRIVEN)
            #include "scenarios/lidDrivenCavity/lidDrivenCavityScenario.cuh"
            using Scenario = LidDrivenScenario;
        // Add your scenario here
        #elif defined(USE_MYNEWSCENARIO)
            #include "scenarios/myNewScenario/myNewScenarioScenario.cuh" // Include your scenario header
            using Scenario = MyNewScenarioScenario; // Set the Scenario type
        #endif
        ```

6.  **Compile and Run:**
    * Recompile the project using your build system (e.g., `make`).
    * Run the simulation executable. It should now use your newly created `MyNewScenario`.


### Code Conventions

host varibles will have name ``h_var``

device variables will have name ``d_var``

# Third-Party libraries

We vendored nanoflann (BSD-licensed) under third_party/nanoflann. See third_party/nanoflann/LICENSE for the full copyright and disclaimer as required by condition (1) and (2) of the BSD license.