# Style Guide
This document outlines guidelines for coding style in this repo.

## Directory structure
The top-level directory should only contain Python files necessary for executing
the main binary (`train.py`).

The **env** directory contains all files necessary for running the simulator,
including the balloon simulator, the wind vector model, the feature vector
constructor, and the gym wrapper.

The **agents** directory contains all files necessary for defining and training
agents which will control the balloon. Note that this may include code necessary
for checkpointing.

The **metrics** directory contains all files necessary for logging and reporting
performance metrics.

## Typing
All variables and functions should be properly typed. We adhere to:
https://docs.python.org/3/library/typing.html

## Member variables and methods
In classes, member variables are either _public_ or _private_. We do not make
use of setters and getters. All private members will have their name prefixed by
`_` and should not be accessed from outside the class.

## Use of `@property` decorator
We discourage the use of `@property` to avoid confusion, unless required to
conform to an external API specification (for example, for Gym). In these cases,
the reason for its use should be documented above the method.

## Abstract classes
We encourage the use of abstract classes to avoid code duplication. Any abstract
class must subclass `abc.ABC` and decorate required methods with
`@abc.abstractmethod`.

## Static methods
Functions that are only called within a class but do not access any class
members (e.g. `self.`) must be made static by decorating them with
`@staticmethod`.

# Data Classes
The `@dataclasses.dataclass` decorator should be used whenever the `__init__`
method would only be setting values based on the constructor parameters.

# Floats
Prefer to use `0.0` over `0.`, as the former is more obviously a float.

## Gin config
We make use of gin config for parameter injection. Rather than passing
parameters from flags all the way to the method/class where it will be used, the
gin-configurable parameters are specified via gin config files (or gin
bindings).

The guidelines for gin-configurable parameters are:
1.  Only set variables in a gin config which have a default value of
    gin.REQUIRED.
1.  Only keyword-only args can be set with a gin config.

For example in the signature below:
```
def f(x: float, y: float = 0.0 *, z: float = 0.0, alpha: float = gin.REQUIRED)
```

the only variable that can (and must) be set via a gin config is `alpha`.

We accept two gin-config files: one for specifying _environment_
parameters (via the `--environment_gin_file` flag), and one for specifying
_agent_ parameters (via the `--agent_gin_file` flag). Any other parameters (or
variations from those specified in the config files) can be specified via
the `--gin_bindings` flags.

For more information see:
https://github.com/google/gin-config
