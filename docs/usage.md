# Usage

A minimal example of importing and using SimKit:

```python
import simkit

mu, lam = simkit.ympr_to_lame(ym=1.0e5, pr=0.45)
print(mu, lam)
```

See the [API reference](autoapi/index) for the full list of available
functions and submodules. Every function's docstring is rendered there
automatically, so writing a good docstring is all you need to do to publish
documentation for a new function.

## Tutorials

Step-by-step tutorial notebooks live in the
[simkit-tutorials](https://github.com/otmanon/simkit-tutorials) repository and
are rendered under the **Tutorials** section of this site. They are kept in a
separate repo so cloning `simkit` stays lightweight.

## Examples

The `examples/` directory in the repository contains runnable scripts that
demonstrate common workflows.
