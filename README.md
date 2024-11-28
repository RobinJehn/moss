# MoSS CW2 Group I: The Sustainability of Nuclear Energy and its Role in the Energy Transition
## Folder structure 
data: Includes the data.csv file, which is used to initialise the Market & Production submodel. The file contains a column fo hourly demand in GW, a column per producer for their hourly capacity in GW (Nuclear,Solar,Wind,Biomass,Gas,Hydro,Coal). 
Data were compiled using the IEA Real-Time Electricity Tracker (https://www.iea.org/data-and-statistics/data-tools/real-time-electricity-tracker?from=2024-10-29&to=2024-11-28&category=demand) and for wind (https://windeurope.org/about-wind/daily-wind/electricity-mix).

src: Includes all .py files used for this study. 
- main.py contains the model and the code to execute multiple runs in a loop. It requires as initialisation input files, a subsidies.csv and the data.csv file. After it is ran, it outputs a simulation_results.csv file.

- create_subsidies.py contains code to create a user adjustable grid of size nxn where each point is a mix of direct to producer subsidies distributed to Nuclear and Renewable Energy Source producers. Once it is ran, it outputs a subsidies.csv file.

- analysis.py contains the code used to aggregate quarterly data, create any visualisations and calculate any statistics used. It requires the simulation_results.csv and subsidies.csv files as inputs. Users can adjust code to control which visualisations they would to receive as output.

todo.md was used during development to track tasks, can be ignored but is left in the repository, as evidence.
sources.md does not represent an exhaustive or accurate list of sources used. It was part of the development process and should be ignore but has been left in the repository as evidence.

## Using the model
To use the model, at its default state, requires executing the create_subsidies.py file and then the main.py file. Executing the analysis.py file is necessary for visualisation, but printing can be enabled within the main.py file for quick use or troubleshooting.