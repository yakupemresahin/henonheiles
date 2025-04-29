# Imports
from IPython.display import display, Latex
import sympy
import os
import json
import matplotlib as mpl
import matplotlib.pyplot as plt
import import_ipynb # Note: This might require specific environment setup (like Jupyter)
from datetime import datetime
from mpl_toolkits import mplot3d
import pandas as pd
import numpy as np
from scipy.optimize import fsolve

# Class Definitions
class phaseSpace:

    def __init__(self, dim, order = 2):
        self.dim = dim
        self.order = order
        self.psDim = dim*order

    def setVariables(self, variableString):
        self.variables = variables = variableString.split()
        if len(variables) == self.psDim:
            print('Variables set:', variables)
            for variable in variables:
                # Initialize variable and its velocity counterpart (assuming pairs)
                exec("self.%s = 0" % (variable))
                # This assumes variables come in pairs like x, px, y, py...
                # The v_ prefix might be intended for variational equations, let's refine based on usage
                # It seems 'v_' variables are set in setVariationalPhase, perhaps remove this line?
                # Let's assume setVariationalPhase handles 'v_' initialization.
                # exec("self.v_%s = 0" % (variable)) # Revisit if needed

            self.setPhase(np.zeros(self.psDim)) # Default initial phase
            self.setVariationalPhase() # Default initial variational phase
        else:
            print('Variable number does not match with phase space dimension')

    def printSystem(self):
        display(Latex(f'$---system---$'))
        # Assuming dynamicsDict is set by setDynamics
        if hasattr(self, 'dynamicsDict') and self.dynamicsDict:
             dynamics = list(self.dynamicsDict.values())
             for idx, var in enumerate(self.variables):
                 # Make variable available in local scope for sympify
                 exec("%s = self.%s" % (var, var))
             for idx, val in enumerate(self.variables):
                  try:
                      # Display formatted equation using sympy for latex rendering
                      display(Latex(f'$\dot{{ {str(val)} }} = {sympy.latex(sympy.sympify(dynamics[idx]))}$'))
                  except Exception as e:
                      print(f"Error processing/displaying equation for {val}: {e}")
                      print(f"Original expression: {dynamics[idx]}")
             display(Latex(f'$--------------------$')) # End marker
        else:
             print("Dynamics not set yet.")


    def setDynamics(self, exprDict):
        if len(exprDict) == self.psDim:
            self.dynamicsDict = exprDict
            # self.printSystem() # Optional: print system when set
        else:
            print('Variable number does not match with phase space dimension')

    def setVariationalDynamics(self, exprDict):
         if len(exprDict) == self.psDim:
             self.variationalDynamicsDict = exprDict
             # self.printSystem() # Optional: Print variational system if needed
         else:
             print('Variable number does not match with phase space dimension')

    def setPhase(self, phase):
        if len(phase) == self.psDim:
            self.phase = np.array(phase) # Ensure it's a numpy array
            for idx, var in enumerate(self.variables):
                # Assign value from phase vector to the corresponding instance attribute
                exec("self.%s = %f" % (var, self.phase[idx]))
        else:
             print('Phase vector length does not match phase space dimension')


    def setVariationalPhase(self, norm=None, phase=None):
        # Default initial variational vector (small displacements)
        default_phase = np.array([0.5] * self.psDim) # Example default
        if phase is None:
             phase = default_phase
        else:
             phase = np.array(phase) # Ensure numpy array

        if len(phase) != self.psDim:
             print('Initial variational phase vector length does not match dimension. Using default.')
             phase = default_phase

        # Default norm for normalization
        if norm is None: norm = 1

        # Normalize the initial variational phase vector
        curNorm = np.sqrt(np.sum(np.power(phase, 2)))
        if curNorm == 0:
             print("Warning: Initial variational phase vector has zero norm. Cannot normalize.")
             # Handle this case, e.g., use a default non-zero vector or raise error
             self.variationalPhase = phase # Assign unnormalized or handle differently
             self.vpNorm = 0 # Norm is zero
        else:
             self.variationalPhase = np.multiply(phase, norm / curNorm)
             self.vpNorm = norm # Store the target norm

        # Set individual v_ variable attributes (useful for eval in derivatives)
        for idx, var in enumerate(self.variables):
             # Assumes variational variable names are v_varname
             exec("self.v_%s = %f" % (var, self.variationalPhase[idx]))


    def resetPhase(self, phase):
        # This seems to intend setting the main phase, similar to setPhase
        # The name 'resetPhase' is slightly ambiguous. Let's assume it calls setPhase.
        # Original code: self.phase = np.zeros(len(self.phase)) - this resets to zeros, might not be intended.
        # Let's stick to the behavior of setPhase for consistency.
        self.setPhase(phase)


    def printPhase(self):
        display(Latex(f'$---phase---$'))
        if hasattr(self, 'variables') and hasattr(self, 'phase'):
            for idx, var in enumerate(self.variables):
                 # Fetch the value directly from the phase vector for printing
                 display(Latex(f'${str(var)} = {self.phase[idx]}$'))
            display(Latex(f'$-----------$'))
        else:
            print("Phase or variables not set.")

    def calcDerivatives(self, phase):
        if not hasattr(self, 'dynamicsDict'):
            raise ValueError("Dynamics dictionary not set. Call setDynamics first.")
        if len(phase) != self.psDim:
            raise ValueError("Input phase dimension mismatch.")

        derivatives = np.zeros(self.psDim)
        local_scope = {} # Create a scope for exec/eval

        # Populate scope with current phase values
        for idx, var in enumerate(self.variables):
             local_scope[var] = phase[idx]
             # Also add self to scope if methods need instance attributes
             local_scope['self'] = self

        # Calculate derivatives using expressions from dynamicsDict
        for idx, var in enumerate(self.variables):
             # The original used exec to define dot<var>, then eval'd it.
             # It's safer and clearer to directly eval the expression.
             try:
                 expression = self.dynamicsDict[var]
                 derivatives[idx] = eval(expression, {"np": np, "sympy": sympy}, local_scope) # Provide numpy/sympy if needed
             except Exception as e:
                 print(f"Error evaluating derivative for {var}: {e}")
                 print(f"Expression: {expression}")
                 print(f"Scope: {local_scope}")
                 raise # Re-raise the exception

        return derivatives

    def calcVariationalDerivatives(self, phase_and_variational):
        # Expects a combined vector: [phase, variational_phase]
        if not hasattr(self, 'variationalDynamicsDict'):
            raise ValueError("Variational dynamics dictionary not set.")
        if len(phase_and_variational) != 2 * self.psDim:
             raise ValueError("Input vector dimension mismatch (expected phase + variational phase).")

        phase = phase_and_variational[0:self.psDim]
        variational_phase = phase_and_variational[self.psDim:]

        derivatives = np.zeros(self.psDim)
        local_scope = {} # Scope for eval

        # Populate scope with current phase and variational phase values
        for idx, var in enumerate(self.variables):
             local_scope[var] = phase[idx]
             local_scope[f"v_{var}"] = variational_phase[idx] # Assumes v_ prefix convention
             # Also add self to scope if methods need instance attributes
             local_scope['self'] = self

        # Calculate derivatives using expressions from variationalDynamicsDict
        for idx, var in enumerate(self.variables):
            # Assumes variational dict keys match variable names
             try:
                 expression = self.variationalDynamicsDict[var]
                 # Directly eval the expression for the derivative of the variational variable
                 derivatives[idx] = eval(expression, {"np": np, "sympy": sympy}, local_scope) # Provide numpy/sympy
             except Exception as e:
                 print(f"Error evaluating variational derivative for v_{var}: {e}")
                 print(f"Expression: {expression}")
                 print(f"Scope: {local_scope}")
                 raise # Re-raise

        return derivatives

    # --- Specific Henon-Heiles Methods ---
    # These seem hardcoded for a specific system (likely Henon-Heiles)
    # It's better practice to pass dynamics via setDynamics/setVariationalDynamics

    def calcDerivativesHH(self, phase):
        # Hardcoded Henon-Heiles derivatives (2D)
        # Assumes phase = [x, y, p_x, p_y]
        if len(phase) != 4: raise ValueError("Expected phase [x, y, px, py] for HH")
        derivatives = [
            phase[2],  # dx/dt = px
            phase[3],  # dy/dt = py
            -phase[0] - 2 * phase[0] * phase[1],  # dpx/dt = -x - 2xy
            -phase[1] - phase[0]**2 + phase[1]**2  # dpy/dt = -y - x^2 + y^2
        ]
        return np.array(derivatives)

    def calcDerivativesHH_variational(self, phase_and_variational):
         # Hardcoded Henon-Heiles + Variational derivatives
         # Assumes input = [x, y, px, py, vx, vy, vpx, vpy]
         if len(phase_and_variational) != 8: raise ValueError("Expected 8D vector for HH variational")

         p = phase_and_variational[0:4]  # Phase: x, y, px, py
         vp = phase_and_variational[4:8] # Variational: vx, vy, vpx, vpy

         # Derivatives of the main phase variables (same as calcDerivativesHH)
         derivatives1 = [
             p[2],
             p[3],
             -p[0] - 2 * p[0] * p[1],
             -p[1] - p[0]**2 + p[1]**2
         ]

         # Derivatives of the variational variables (linearized system)
         # dvx/dt = vpx
         # dvy/dt = vpy
         # dvpx/dt = -(1 + 2*y)*vx - (2*x)*vy
         # dvpy/dt = -(2*x)*vx - (1 - 2*y)*vy
         derivatives2 = [
             vp[2], # dvx/dt = vpx
             vp[3], # dvy/dt = vpy
             -(1 + 2 * p[1]) * vp[0] - (2 * p[0]) * vp[1], # dvpx/dt
             -(2 * p[0]) * vp[0] - (1 - 2 * p[1]) * vp[1]  # dvpy/dt, NOTE: fixed sign error from OCR potentially (-1 - (-2y)) -> (-1+2y), but - (1 - 2y) matches standard form
         ]

         # Concatenate phase derivatives and variational derivatives
         return np.concatenate((derivatives1, derivatives2))


    def setHamiltonian(self, expr):
        # Store the Hamiltonian expression as a string
        self.Hexpr = expr
        # Try to display it using Sympy for nice LaTeX output
        try:
            display(Latex(f'$H = {sympy.latex(sympy.sympify(expr))}$'))
            # Initialize energy value to 0? Or calculate it? Let's just store expr.
            # The original exec("self.H = 0") is confusing. Let's calculate H instead.
            # self.H = self.calcEnergy() # Calculate initial energy if phase is set
        except Exception as e:
             print(f"Could not display Hamiltonian using Sympy: {e}")
             print(f"Hamiltonian expression stored: {self.Hexpr}")


    def calcEnergy(self, phase=None):
        if not hasattr(self, 'Hexpr'):
            raise ValueError("Hamiltonian expression not set. Call setHamiltonian first.")

        if phase is None:
            if not hasattr(self, 'phase'):
                 raise ValueError("Phase not set and not provided.")
            phase = self.phase
        else:
            phase = np.array(phase) # Ensure numpy array

        if len(phase) != self.psDim:
             raise ValueError("Input phase dimension mismatch.")

        local_scope = {} # Scope for eval
        # Populate scope with current phase values
        for idx, var in enumerate(self.variables):
             local_scope[var] = phase[idx]
             # Add self if needed
             local_scope['self'] = self

        # Evaluate the stored Hamiltonian expression string
        try:
            # Provide math functions if needed, e.g., from numpy or math
            energy = eval(self.Hexpr, {"np": np, "sympy": sympy}, local_scope)
            return float(energy) # Ensure it's a float
        except Exception as e:
             print(f"Error evaluating Hamiltonian: {e}")
             print(f"Expression: {self.Hexpr}")
             print(f"Scope: {local_scope}")
             raise # Re-raise


    def setPhaseWithEnergy(self, phase, energy):
        # Finds a value for a 'None' entry in phase to match the target energy.
        if None not in phase:
            print("Phase vector does not contain 'None'. Cannot adjust for energy.")
            current_energy = self.calcEnergy(phase)
            print(f"Current energy: {current_energy}, Target energy: {energy}")
            if np.isclose(current_energy, energy):
                 self.setPhase(phase)
                 self.strIC = str(phase) + '_' + str(energy) # Store IC representation
            else:
                 print("Phase set, but energy does not match target.")
            return

        varId = phase.index(None) # Find the index of the variable to adjust

        # Define the cost function for the solver: difference from target energy
        def energyCost(a):
            # 'a' is the value being solved for (usually a list/array for fsolve)
            testPhase = list(phase) # Create a mutable copy
            testPhase[varId] = a[0] # Substitute the trial value
            # Calculate energy for this trial phase
            # Need to handle potential errors during eval if 'a[0]' is invalid
            try:
                current_energy = self.calcEnergy(testPhase)
                return current_energy - energy
            except Exception as e:
                # If eval fails (e.g., invalid intermediate value), return a large error
                print(f"Warning: Error during energy calculation in solver: {e}")
                return 1e10 # Return large number

        # Initial guess for the unknown variable (e.g., 1 or 0)
        initial_guess = [1.0] # fsolve expects an array-like object

        # Use fsolve to find the root of energyCost (where energyCost is zero)
        try:
            res, infodict, ier, mesg = fsolve(energyCost, initial_guess, xtol=1e-7, full_output=True) # Adjust tolerance

            if ier == 1: # Check if solver succeeded
                newPhase = list(phase)
                newPhase[varId] = res[0]
                final_energy = self.calcEnergy(newPhase)

                # Verify the result
                if np.isclose(final_energy, energy, atol=1e-5): # Use tolerance for check
                     print(f"Successfully found phase for energy {energy}.")
                     self.setPhase(newPhase)
                     # Store a string representation of the initial condition
                     self.strIC = str(self.phase.tolist()) + '_' + str(energy)
                else:
                     print(f'Solver finished, but energy mismatch: {final_energy} vs {energy}')
                     print(f"Solver message: {mesg}")
            else:
                # Solver failed
                print(f'Solver failed to find phase for energy {energy}.')
                print(f"Solver message: {mesg}")

        except Exception as e:
            print(f"Error during solving process: {e}")


class timeEvolution:

    def __init__(self, system: phaseSpace): # Type hint for clarity
        self.system = system
        # Initialize attributes that will be set later
        self.t = None
        self.dt = None
        self.strIC = getattr(system, 'strIC', 'unknown_IC') # Get IC string if set
        self.vpNorm = getattr(system, 'vpNorm', None) # Get vpNorm if set
        self.timeAxis = None
        self.phaseAxis = None
        self.variationalPhaseAxis = None
        self.energy = None # Store energy if relevant


    def RK4Step(self, phase, dt):
        # Standard RK4 step using the system's derivative function
        k1 = self.system.calcDerivativesHH(phase) # Using HH version directly based on context
        k2 = self.system.calcDerivativesHH(phase + 0.5 * dt * k1)
        k3 = self.system.calcDerivativesHH(phase + 0.5 * dt * k2)
        k4 = self.system.calcDerivativesHH(phase + dt * k3)
        y_nn = phase + (1/6) * (k1 + 2*k2 + 2*k3 + k4) * dt
        return y_nn

    def RK4StepWithVariational(self, phase_and_variational, dt):
        # RK4 step for the combined phase + variational system
        # Uses the combined derivative function
        k1 = self.system.calcDerivativesHH_variational(phase_and_variational)
        k2 = self.system.calcDerivativesHH_variational(phase_and_variational + 0.5 * dt * k1)
        k3 = self.system.calcDerivativesHH_variational(phase_and_variational + 0.5 * dt * k2)
        k4 = self.system.calcDerivativesHH_variational(phase_and_variational + dt * k3)
        y_nn = phase_and_variational + (1/6) * (k1 + 2*k2 + 2*k3 + k4) * dt
        return y_nn


    def evolveRK4(self, t, dt, IC = None):
        self.t = t
        self.dt = dt

        if IC is None:
            if hasattr(self.system, 'phase'):
                IC = self.system.phase
            else:
                raise ValueError("Initial Condition (IC) not provided and not set in system.")
        else:
             # If IC is provided, set it in the system object too
             self.system.setPhase(IC)

        # Ensure variational phase is initialized in the system object
        if not hasattr(self.system, 'variationalPhase'):
             print("Warning: Variational phase not explicitly set in system. Initializing default.")
             self.system.setVariationalPhase() # Initialize with defaults if needed
        initial_variational_phase = self.system.variationalPhase
        self.vpNorm = self.system.vpNorm # Store the norm used

        # Store IC string representation if available from system setup
        self.strIC = getattr(self.system, 'strIC', str(IC.tolist())) # Fallback to simple list string

        # --- Time evolution ---
        start_time = time.time()

        num_steps = int(np.ceil(t / dt)) # Use ceil to ensure total time t is covered
        self.timeAxis = np.linspace(0, num_steps * dt, num_steps + 1) # Adjusted time axis
        actual_t = num_steps * dt # Actual end time
        if not np.isclose(actual_t, t):
            print(f"Note: Simulation ran for {actual_t:.4f} to accommodate integer steps (target t={t:.4f})")


        # Initialize arrays to store results
        self.phaseAxis = np.zeros((len(self.timeAxis), len(IC)))
        self.variationalPhaseAxis = np.zeros((len(self.timeAxis), len(initial_variational_phase)))

        # Set initial conditions
        self.phaseAxis[0] = IC
        self.variationalPhaseAxis[0] = initial_variational_phase

        # Initial combined state vector
        current_state = np.concatenate((IC, initial_variational_phase))

        # Renormalization constant for variational vector (if vpNorm is set)
        renorm_factor = 1.0 # Default if no normalization needed
        if self.vpNorm is not None and self.vpNorm > 0:
             # We need to re-normalize the variational part at each step (or periodically)
             # The original code seems to apply normalization *before* the step using the norm from the *previous* step's variational vector.
             # This maintains the magnitude specified by vpNorm.
             pass # Renormalization logic applied inside the loop


        # Integration loop
        for i in range(num_steps):
             # --- Renormalization (Based on original code's apparent logic) ---
             # Normalize the *current* variational part before concatenating for the next step's input.
             # This ensures the variational vector used in RK4StepWithVariational starts with the correct norm.
            current_variational = current_state[self.system.psDim:]
            current_norm = np.linalg.norm(current_variational)

            if self.vpNorm is not None and self.vpNorm > 0 and current_norm > 0:
                renorm_factor = self.vpNorm / current_norm
                normalized_variational = current_variational * renorm_factor
            else:
                normalized_variational = current_variational # Use as is if norm is zero or vpNorm not set

            # State vector with potentially renormalized variational part for input to RK4
            input_state = np.concatenate((current_state[:self.system.psDim], normalized_variational))

            # Perform RK4 step
            # Use the combined RK4 step function
            current_state = self.RK4StepWithVariational(input_state, self.dt)

            # Store results for this time step
            self.phaseAxis[i+1] = current_state[0:self.system.psDim]
            self.variationalPhaseAxis[i+1] = current_state[self.system.psDim:]


        # Update the system's final state
        self.system.setPhase(self.phaseAxis[-1])
        self.system.variationalPhase = self.variationalPhaseAxis[-1] # Update variational phase too

        end_time = time.time()
        print(f'Time elapsed: {end_time - start_time:.2f} s')


    def saveTimeEvolution(self, folder):
        if self.timeAxis is None or self.phaseAxis is None or self.variationalPhaseAxis is None:
             print("Error: Simulation data not available. Run evolveRK4 first.")
             return

        # Create folder if it doesn't exist
        if not os.path.exists(folder):
            os.makedirs(folder)
            print(f"Created directory: {folder}")

        # Generate timestamp string
        now = datetime.now()
        dt_string = now.strftime("_%d_%m_%Y__%H_%M_%S") # Date-time format

        # Construct filename
        # Using the stored strIC for uniqueness based on initial conditions
        base_filename = f'henonHeilesSim_t:{self.t:.6g}_dt:{self.dt:.6g}_IC:{self.strIC}'
        # Clean filename: replace potential problematic chars like '[', ']', ',' spaces with '_'
        safe_base_filename = base_filename.replace('[','').replace(']','').replace(', ','_').replace(' ','')
        fileName = os.path.join(folder, f"{safe_base_filename}{dt_string}.csv")


        # Prepare data for saving
        # Combine time, phase, and variational phase into one array
        data_to_save = np.column_stack((self.timeAxis, self.phaseAxis, self.variationalPhaseAxis))

        # Define column names based on system variables
        phase_vars = self.system.variables
        variational_vars = ['v_' + var for var in phase_vars]
        column_values = ['time'] + phase_vars + variational_vars

        # Create Pandas DataFrame for easy CSV writing with headers
        df = pd.DataFrame(data = data_to_save, columns = column_values)

        # Save to CSV
        try:
            with open(fileName, 'w') as fout:
                df.to_csv(fout, index=False, lineterminator='\n') # Use index=False, standard newline
            print(f"Saved simulation data to: {fileName}")
        except Exception as e:
            print(f"Error saving data to {fileName}: {e}")


    def loadTimeEvolution(self, t, dt, IC, energy, folder='/simOuts'): # Default folder added
        # Construct target filename pattern elements based on inputs
        target_t_str = f"t:{t:.6g}"
        target_dt_str = f"dt:{dt:.6g}"
        # Recreate the expected IC string format used during saving
        target_ic_str = f"IC:{str(list(IC)).replace('[','').replace(']','').replace(', ','_').replace(' ','')}" # Match cleaning
        # Note: Energy matching wasn't explicitly in the original save filename format,
        # but the original load code parsed 'fn[4]' which implies it might have been intended or added elsewhere.
        # Let's assume filename includes IC but maybe not energy explicitly for matching.

        search_path = os.path.join(os.getcwd(), folder.lstrip('/')) # Construct full path

        if not os.path.exists(search_path):
            print(f"Directory not found: {search_path}")
            return False # Indicate failure

        print(f"Searching for file matching: t={t}, dt={dt}, IC={IC} in {search_path}")

        found_file = None
        try:
            files = os.listdir(search_path)
            for fileName in files:
                # Check if filename contains the required parts
                if (target_t_str in fileName and
                    target_dt_str in fileName and
                    target_ic_str in fileName and
                    fileName.endswith('.csv')):
                    # Basic check passed, could refine parsing if needed (like the OCR code tried)
                    print(f"Found potential match: {fileName}")
                    found_file = os.path.join(search_path, fileName)
                    break # Take the first match

        except FileNotFoundError:
            print(f"Directory not found during listdir: {search_path}")
            return False
        except Exception as e:
            print(f"Error reading directory {search_path}: {e}")
            return False


        if found_file:
            print(f"Loading data from: {found_file}")
            try:
                df = pd.read_csv(found_file)

                # Extract data into numpy arrays
                self.timeAxis = df['time'].to_numpy()

                phase_vars = self.system.variables
                self.phaseAxis = df[phase_vars].to_numpy()

                variational_vars = ['v_' + var for var in phase_vars]
                self.variationalPhaseAxis = df[variational_vars].to_numpy()

                # Set internal state variables from loaded data/inputs
                self.t = t # Store target t
                self.dt = dt # Store target dt
                self.system.setPhase(self.phaseAxis[-1]) # Set system to final state
                self.system.variationalPhase = self.variationalPhaseAxis[-1]
                self.strIC = str(list(IC)) # Store the IC used for loading
                self.energy = energy # Store the energy associated with this run
                self.vpNorm = getattr(self.system, 'vpNorm', None) # Try to retrieve from system

                print("Successfully loaded and parsed data.")
                return True # Indicate success

            except FileNotFoundError:
                 print(f"Error: File disappeared before reading: {found_file}")
                 return False
            except KeyError as e:
                 print(f"Error: Column not found in CSV file {found_file}. Mismatched data? Missing column: {e}")
                 # Reset loaded data on error
                 self.timeAxis = None
                 self.phaseAxis = None
                 self.variationalPhaseAxis = None
                 return False
            except Exception as e:
                 print(f"Error reading or parsing CSV file {found_file}: {e}")
                 # Reset loaded data on error
                 self.timeAxis = None
                 self.phaseAxis = None
                 self.variationalPhaseAxis = None
                 return False
        else:
            print(f"File not found matching criteria.")
            return False # Indicate failure


    # --- Plotting Methods ---

    def setPlotTime(self, plotTime):
        # Finds the index corresponding to the closest time in timeAxis
        if self.timeAxis is None:
             print("Error: Time axis not available. Run simulation or load data first.")
             return
        self.plotTime = plotTime
        # Find index of the time value closest to plotTime
        self.timeIdx = np.argmin(np.abs(self.timeAxis - plotTime))
        print(f"Plotting up to time index {self.timeIdx} (time ~ {self.timeAxis[self.timeIdx]:.3f})")


    def plot(self, var, colored = False):
        if self.phaseAxis is None or self.timeAxis is None:
             print("Error: Simulation data not available.")
             return
        if not hasattr(self, 'timeIdx'):
            print("Plot time index not set. Call setPlotTime first or plotting full range.")
            time_idx_to_plot = len(self.timeAxis) # Plot everything
        else:
            time_idx_to_plot = self.timeIdx + 1 # Include the index itself

        try:
            varId = self.system.variables.index(var)
        except ValueError:
            print(f"Error: Variable '{var}' not found in system variables: {self.system.variables}")
            return
        except AttributeError:
             print("Error: System object or its variables list is not defined.")
             return

        varAxis = self.phaseAxis[:time_idx_to_plot, varId]
        time_axis_to_plot = self.timeAxis[:time_idx_to_plot]

        plt.figure()
        plt.plot(time_axis_to_plot, varAxis, c='black', alpha=0.7, label=f'{var}(t)') # Label added

        if colored:
            # Use time itself for color mapping
            colors = time_axis_to_plot
            scatter = plt.scatter(time_axis_to_plot, varAxis, c=colors, cmap='viridis', s=10, label='Time (color)') # Viridis cmap
            plt.colorbar(scatter, label='Time (s)')

        plt.xlabel('Time (s)')
        plt.ylabel(f'${var}$') # Use LaTeX formatting for variable name
        plt.title(f'Time Evolution of {var}')
        plt.grid(True, linestyle='--', alpha=0.6)
        plt.legend()
        # plt.show() # Add this if not running in interactive environment like Jupyter


    def plot2D(self, var1, var2, colored = False):
        if self.phaseAxis is None or self.timeAxis is None:
             print("Error: Simulation data not available.")
             return
        if not hasattr(self, 'timeIdx'):
            print("Plot time index not set. Call setPlotTime first or plotting full range.")
            time_idx_to_plot = len(self.timeAxis)
        else:
            time_idx_to_plot = self.timeIdx + 1

        try:
            varId1 = self.system.variables.index(var1)
            varId2 = self.system.variables.index(var2)
        except ValueError as e:
            print(f"Error: Variable not found: {e}")
            return
        except AttributeError:
             print("Error: System object or its variables list is not defined.")
             return


        varAxis1 = self.phaseAxis[:time_idx_to_plot, varId1]
        varAxis2 = self.phaseAxis[:time_idx_to_plot, varId2]
        time_axis_to_plot = self.timeAxis[:time_idx_to_plot]

        plt.figure(figsize=(8, 6)) # Slightly larger figure for 2D plot
        plt.plot(varAxis1, varAxis2, c='black', alpha=0.5, label='Trajectory') # Basic trajectory

        if colored:
            # Use time for color mapping
            colors = time_axis_to_plot
            scatter = plt.scatter(varAxis1, varAxis2, c=colors, cmap='viridis', s=10, label='Time (color)')
            plt.colorbar(scatter, label='Time (s)')

        plt.xlabel(f'${var1}$') # LaTeX label
        plt.ylabel(f'${var2}$') # LaTeX label
        plt.title(f'Phase Space Plot: {var2} vs {var1}')
        plt.grid(True, linestyle='--', alpha=0.6)
        plt.axis('equal') # Often useful for phase space plots
        plt.legend()
        # plt.show()


    def plot3D(self, var1, var2, var3, colored = False):
         if self.phaseAxis is None or self.timeAxis is None:
             print("Error: Simulation data not available.")
             return
         if not hasattr(self, 'timeIdx'):
             print("Plot time index not set. Call setPlotTime first or plotting full range.")
             time_idx_to_plot = len(self.timeAxis)
         else:
             time_idx_to_plot = self.timeIdx + 1

         try:
             varId1 = self.system.variables.index(var1)
             varId2 = self.system.variables.index(var2)
             varId3 = self.system.variables.index(var3)
         except ValueError as e:
             print(f"Error: Variable not found: {e}")
             return
         except AttributeError:
             print("Error: System object or its variables list is not defined.")
             return

         varAxis1 = self.phaseAxis[:time_idx_to_plot, varId1]
         varAxis2 = self.phaseAxis[:time_idx_to_plot, varId2]
         varAxis3 = self.phaseAxis[:time_idx_to_plot, varId3]
         time_axis_to_plot = self.timeAxis[:time_idx_to_plot]

         fig = plt.figure(figsize=(10, 8)) # Larger figure for 3D
         ax = fig.add_subplot(111, projection='3d') # Correct way to get 3D axes

         # Plot the trajectory line
         ax.plot(varAxis1, varAxis2, varAxis3, c='black', alpha=0.5, label='Trajectory')

         if colored:
             # Use time for color mapping on a scatter plot overlay
             colors = time_axis_to_plot
             scatter = ax.scatter(varAxis1, varAxis2, varAxis3, c=colors, cmap='viridis', s=10, label='Time (color)')
             fig.colorbar(scatter, ax=ax, label='Time (s)', shrink=0.6) # Add color bar to the figure

         ax.set_xlabel(f'${var1}$')
         ax.set_ylabel(f'${var2}$')
         ax.set_zlabel(f'${var3}$')
         ax.set_title(f'3D Phase Space Plot: {var1}, {var2}, {var3}')
         ax.grid(True)
         # You might need to adjust view angle: ax.view_init(elev=20., azim=-35)
         # plt.show()


    def poincareSurface(self, var1, var2, poincareVar, poincareVal=0.0, positiveCrossings=True, clr='black'):
         # Generates a Poincare section plot
         if self.phaseAxis is None or self.timeAxis is None:
             print("Error: Simulation data not available.")
             return
         # No timeIdx needed here, use full trajectory

         try:
             varId1 = self.system.variables.index(var1)
             varId2 = self.system.variables.index(var2)
             varId3 = self.system.variables.index(poincareVar) # The variable defining the surface
         except ValueError as e:
             print(f"Error: Variable not found: {e}")
             return
         except AttributeError:
             print("Error: System object or its variables list is not defined.")
             return

         # Get the full trajectories
         varAxis1 = self.phaseAxis[:, varId1]
         varAxis2 = self.phaseAxis[:, varId2]
         poincareAxis = self.phaseAxis[:, varId3]

         # Find points where the trajectory crosses the Poincare surface
         # Crossing occurs when poincareAxis - poincareVal changes sign.
         crossings = np.where(np.diff(np.sign(poincareAxis - poincareVal)))[0]

         # Filter for positive or negative direction crossings if needed
         if positiveCrossings:
              # Crossing from negative to positive: sign change from -1 to 1 (or 0 to 1)
              # Requires poincareAxis[idx] < poincareVal and poincareAxis[idx+1] >= poincareVal
               crossing_indices = [idx for idx in crossings if poincareAxis[idx] < poincareVal and poincareAxis[idx+1] >= poincareVal]
         else: # Negative crossings
              # Crossing from positive to negative: sign change from 1 to -1 (or 0 to -1)
              # Requires poincareAxis[idx] > poincareVal and poincareAxis[idx+1] <= poincareVal
               crossing_indices = [idx for idx in crossings if poincareAxis[idx] > poincareVal and poincareAxis[idx+1] <= poincareVal]

         if not crossing_indices:
              print(f"No {'positive' if positiveCrossings else 'negative'} crossings found for {poincareVar} = {poincareVal}.")
              return

         # Interpolate to find the precise values of var1 and var2 at the crossing
         poincare_var1 = []
         poincare_var2 = []

         for idx in crossing_indices:
             # Linear interpolation
             # Find fraction t where crossing occurs between idx and idx+1
             val_before = poincareAxis[idx] - poincareVal
             val_after = poincareAxis[idx+1] - poincareVal

             if np.isclose(val_after - val_before, 0): # Avoid division by zero if values are identical
                  t_interp = 0.5 # Or handle as appropriate, maybe skip?
             else:
                  t_interp = -val_before / (val_after - val_before)

             # Clamp interpolation factor to [0, 1] just in case
             t_interp = np.clip(t_interp, 0.0, 1.0)

             # Interpolate var1 and var2
             interp_var1 = varAxis1[idx] + t_interp * (varAxis1[idx+1] - varAxis1[idx])
             interp_var2 = varAxis2[idx] + t_interp * (varAxis2[idx+1] - varAxis2[idx])

             poincare_var1.append(interp_var1)
             poincare_var2.append(interp_var2)

         # Plot the Poincare section
         plt.figure(figsize=(8, 8)) # Square figure often good for Poincare
         plt.scatter(poincare_var1, poincare_var2, color=clr, marker='.', s=10) # Smaller markers
         plt.xlabel(f'${var1}$')
         plt.ylabel(f'${var2}$')
         crossing_direction = "positive" if positiveCrossings else "negative"
         plt.title(f'Poincaré Section ({poincareVar} = {poincareVal}, {crossing_direction} crossings)')
         plt.grid(True, linestyle='--', alpha=0.6)
         plt.axis('equal')
         # plt.show()


# --- Example Usage / Simulation Setup ---
if __name__ == "__main__": # Ensures this runs only when script is executed directly

    # 1. Initialize the phase space system
    henonHeiles = phaseSpace(dim=2, order=2) # 2 dimensions (x,y), order 2 (pos, vel) -> psDim=4

    # 2. Define variables
    henonHeiles.setVariables('x y p_x p_y')

    # 3. Set the dynamics (equations of motion)
    henonHeiles.setDynamics({
        'x': 'p_x',
        'y': 'p_y',
        'p_x': '-x - 2*x*y',
        'p_y': '-y - x**2 + y**2'
    })

    # 4. Set the variational dynamics (linearized equations)
    # Note: Keys should match base variables (x, y, px, py)
    # Expressions should yield derivatives of the corresponding *variational* variable (vx, vy, vpx, vpy)
    henonHeiles.setVariationalDynamics({
        'x':   'v_p_x',                                  # dvx/dt = vpx
        'y':   'v_p_y',                                  # dvy/dt = vpy
        'p_x': '-(1 + 2*y)*v_x - (2*x)*v_y',             # dvpx/dt
        'p_y': '-(2*x)*v_x - (1 - 2*y)*v_y'              # dvpy/dt (using - (1 - 2y) form)
    })


    # 5. Set the Hamiltonian
    H_expr = "(1/2)*(p_x**2 + p_y**2) + (1/2)*(x**2 + y**2) + x**2*y - (1/3)*y**3"
    henonHeiles.setHamiltonian(H_expr)

    # 6. Define simulation parameters
    energy = 0.125
    t_total = 1000 # Shorter time for quick test
    dt = 0.01    # Time step

    # 7. Set initial conditions using energy constraint
    # Example: Fix x=0, y=0.1, px=None, py=0, find px for given energy
    # ic_template = [0.0, 0.2, None, 0.0]
    # henonHeiles.setPhaseWithEnergy(ic_template, energy) # This calculates and sets the initial phase

    # Or define multiple initial conditions
    initial_conditions = [
         # [x,   y,    px,      py]
         [0.0, 0.1,   None,    0.0],
         [0.0, 0.2,   None,    0.0],
         [0.0, 0.3,   None,    0.0],
         [0.0, -0.15, None,    0.0],
         [0.0, -0.25, None,    0.0],
         [0.0, -0.35, None,    0.0]
    ]

    simulations = [] # Store simulation objects

    # 8. Run simulations for each initial condition
    output_folder = 'simOuts_example' # Define output directory
    plt.figure(figsize=(8, 8)) # Create one figure for Poincare plot

    for i, ic_template in enumerate(initial_conditions):
        print(f"\n--- Running Simulation {i+1} ---")
        # Find phase for energy and set it in the henonHeiles object for this run
        henonHeiles.setPhaseWithEnergy(ic_template, energy)

        # Check if phase setting was successful
        if not hasattr(henonHeiles, 'phase'):
            print(f"Skipping simulation {i+1} due to phase setting error.")
            continue

        print(f"Starting IC {i+1}: {henonHeiles.phase.tolist()}")

        # Create a new timeEvolution instance for each simulation
        sim = timeEvolution(henonHeiles) # Pass the system with the IC set

        # Evolve the system
        sim.evolveRK4(t_total, dt) # IC is now taken from the system object

        # Save results (optional)
        sim.saveTimeEvolution(output_folder)

        # Add to list (optional)
        simulations.append(sim)

        # Generate and add Poincare points to the common plot
        # Example: Poincare section for x=0, plotting p_y vs y
        sim.poincareSurface('y', 'p_y', 'x', poincareVal=0.0, positiveCrossings=True, clr=plt.cm.viridis(i / len(initial_conditions)))


    # Finalize and show the combined Poincare plot
    plt.xlabel('$y$')
    plt.ylabel('$p_y$')
    plt.title(f'Poincaré Sections (x=0, E={energy})')
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.axis('equal')
    plt.show() # Display the plot

    # --- Example of loading and plotting one simulation ---
    if simulations: # If at least one simulation ran
        print("\n--- Loading and Plotting First Simulation ---")
        sim_to_load = simulations[0]
        # Use the parameters from the first simulation to load it back
        loaded_ok = sim_to_load.loadTimeEvolution(
            sim_to_load.t,
            sim_to_load.dt,
            sim_to_load.phaseAxis[0,:4], # Get original IC from stored data
            sim_to_load.energy if hasattr(sim_to_load, 'energy') else energy, # Get stored energy or fallback
            folder=output_folder.lstrip('/') # Pass the relative folder name
        )

        if loaded_ok:
            sim_to_load.setPlotTime(t_total / 2) # Plot up to half the time
            sim_to_load.plot('y', colored=True)
            sim_to_load.plot2D('x', 'p_x', colored=True)
            sim_to_load.plot3D('x', 'y', 'p_x', colored=True)
            plt.show() # Show all plots generated