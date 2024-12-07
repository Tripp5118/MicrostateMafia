# MicrostateMafia

**Contributors**: Joshua Camacho, Nicolas Flores, Elias Martin, Robert Robinson, Andye Tosh

## To Run This Code

You will need to install the following Python libraries:

```bash
pip install matplotlib scipy numpy
```

We think the `.zip` we’ll submit might include a `.venv` (virtual environment) already set up. 

The exact data we collected is stored in the file `results-thermo-project.xlsx`. All scripts will output pictures to the root directory of the project.

There is currently a bug with Matplotlib: when the code opens the GIF of the phase diagram and GX curves (along with the GX curves at specific temperatures), the program will hang unless you close the GIF window before closing the GX curve image. 
