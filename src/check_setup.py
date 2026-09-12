import sys

import pandas
import numpy
import sklearn
import pydantic


def main() -> None:
    print("Python:", sys.version)
    print("Pandas:", pandas.__version__)
    print("NumPy:", numpy.__version__)
    print("Scikit-learn:", sklearn.__version__)
    print("Pydantic:", pydantic.__version__)
    print("\nEnvironment setup successful!")


if __name__ == "__main__":
    main()