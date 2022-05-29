# PhD Utils Scripts
A set of python scripts to use for initial exploration and analysis and several utils for speeding up the tests.

---

## Requirements
At the current state, the requirements are based on numpy and partially on opencv. There is also a part in blender (bpy) for rendering, which is still in early stage of development. The full list of requirements can be found in the `requirements.txt` file.

## Installation:
To install it locally:

`python setup.py install`

## Development contrib
For development you can build and then link the code using the -e command

```
python setup.py bdist_wheel

pip install -e .
```

For Wheel only mode:

`python setup.py bdist_wheel`

---

### Thanks
Part of the code was borrowed from MCarletti who was my Co-Advisor during my Master's, in particular regarding the Blender part. 
