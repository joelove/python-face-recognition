# python-face-recognition
## A real-time facial recognition tool built with Python

### Prerequisites

#### [Python](https://docs.python-guide.org/starting/install3/osx/)

Mac OS X comes with Python 2.7 out of the box. However, this project uses Python 3.7 so we'll install it using [Homebrew](https://brew.sh/):

```shell
brew install python
```

You'll probably need to augment your PATH in `.bash_profile` or `.zshrc`, too:

```shell
export PATH="/usr/local/opt/python/libexec/bin:$PATH"
```

If you're using Linux or Windows, you'll need to work this bit out yourself.

#### [Poetry](https://poetry.eustace.io/docs/)

This project uses [Poetry](https://poetry.eustace.io/docs/) to manage packages and environments. Frankly, I couldn't find a viable alternative amongst the vast menagerie of whacky and incomplete Python package managers. [PyPI](https://pypi.org/) is still terrible so packages aren't any easier to find but at least we won't need to fight with Pip and Virtualenv:

```shell
curl -sSL https://raw.githubusercontent.com/sdispater/poetry/master/get-poetry.py | python
```

You still might need to install [Pip](https://pypi.org/project/pip/) and [Virtualenv](https://virtualenv.pypa.io/en/latest/) so Poetry can use them behind the scenes.

### Installation

If you're running MacOS and the many malevolent gods of the Python ecosystem are smiling upon you today, you should just be able to install the dependencies using Poetry:

```shell
poetry install
```

### Training the script

Before running the script, save a .jpg for each face you want to match in the `faces` directory. Training will happen automatically before the first run and the name of the file will be the identifier for that face.

```shell
/faces
├── john-doe.jpg
└── jenna-bloggs.jpg
```

If you want to retrain the script with new faces just delete `faces/faces.npy`:

```shell
rm faces/faces.npy
```

### Running the script

If you're holding a lucky rabbit's foot and pigs have been flying in your little corner of the infinite multiverse since the emergence of the cosmos, you can just run:

```shell
poetry run python main.py
```

### Credit

This tool is based on several different tools and is made possible in no small part by the hard work of engineers much more talented than myself, including but not limited to:

- [Davis E. King](https://github.com/davisking/dlib-models)
- [Mikko Pohja](https://github.com/mikko/laervi)

---

[![Sponsored](https://img.shields.io/badge/chilicorn-sponsored-brightgreen.svg?logo=data%3Aimage%2Fpng%3Bbase64%2CiVBORw0KGgoAAAANSUhEUgAAAA4AAAAPCAMAAADjyg5GAAABqlBMVEUAAAAzmTM3pEn%2FSTGhVSY4ZD43STdOXk5lSGAyhz41iz8xkz2HUCWFFhTFFRUzZDvbIB00Zzoyfj9zlHY0ZzmMfY0ydT0zjj92l3qjeR3dNSkoZp4ykEAzjT8ylUBlgj0yiT0ymECkwKjWqAyjuqcghpUykD%2BUQCKoQyAHb%2BgylkAyl0EynkEzmkA0mUA3mj86oUg7oUo8n0k%2FS%2Bw%2Fo0xBnE5BpU9Br0ZKo1ZLmFZOjEhesGljuzllqW50tH14aS14qm17mX9%2Bx4GAgUCEx02JySqOvpSXvI%2BYvp2orqmpzeGrQh%2Bsr6yssa2ttK6v0bKxMBy01bm4zLu5yry7yb29x77BzMPCxsLEzMXFxsXGx8fI3PLJ08vKysrKy8rL2s3MzczOH8LR0dHW19bX19fZ2dna2trc3Nzd3d3d3t3f39%2FgtZTg4ODi4uLj4%2BPlGxLl5eXm5ubnRzPn5%2Bfo6Ojp6enqfmzq6urr6%2Bvt7e3t7u3uDwvugwbu7u7v6Obv8fDz8%2FP09PT2igP29vb4%2BPj6y376%2Bu%2F7%2Bfv9%2Ff39%2Fv3%2BkAH%2FAwf%2FtwD%2F9wCyh1KfAAAAKXRSTlMABQ4VGykqLjVCTVNgdXuHj5Kaq62vt77ExNPX2%2Bju8vX6%2Bvr7%2FP7%2B%2FiiUMfUAAADTSURBVAjXBcFRTsIwHAfgX%2FtvOyjdYDUsRkFjTIwkPvjiOTyX9%2FAIJt7BF570BopEdHOOstHS%2BX0s439RGwnfuB5gSFOZAgDqjQOBivtGkCc7j%2B2e8XNzefWSu%2BsZUD1QfoTq0y6mZsUSvIkRoGYnHu6Yc63pDCjiSNE2kYLdCUAWVmK4zsxzO%2BQQFxNs5b479NHXopkbWX9U3PAwWAVSY%2FpZf1udQ7rfUpQ1CzurDPpwo16Ff2cMWjuFHX9qCV0Y0Ok4Jvh63IABUNnktl%2B6sgP%2BARIxSrT%2FMhLlAAAAAElFTkSuQmCC)](http://spiceprogram.org/oss-sponsorship)
