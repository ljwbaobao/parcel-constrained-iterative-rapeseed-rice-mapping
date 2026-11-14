"""Run source-region iterative training and target-region majority-vote classification."""

from importlib import import_module

run = import_module("6_Transfer_Iterative_RF_Core").run


if __name__ == "__main__":
    run()
