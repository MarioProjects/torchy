# **Torchy** - A Pytorch library with some helper functions

### Requisitos

Para la utilización de esta 'librería' es necesaria la instalación de algunas librerías de Python. Se recomienda la utilización de entornos virtuales:

```sh
cd ~
virtualenv -p python3.5 venv
source venv/bin/activate

python3 -m pip install numpy pandas scipy matplotlib sklearn scikit-learn scikit-image slackclient cmake
python3 -m pip install tqdm opencv-python flask dlib cython seaborn
python3 -m pip install albumentations
python3 -m pip install ipython prompt-toolkit
python3 -m pip install kaggle --upgrade
```

A continuación la instalación de la 'librería' en sí es tan sencilla como:
```sh
pip install git+https://github.com/MarioProjects/torchy.git
```

Por último, si deseamos utilizar la funcionalidad de Slack ([para hacer el logging más fácil](https://github.com/MarioProjects/Python-Slack-Logging)) deberemos añadir al sistema el [token](https://github.com/MarioProjects/Python-Slack-Logging) de nuestro espacio de trabajo al sistema:

```sh
echo "export SLACK_TOKEN='my-slack-token'" >> ~/.bashrc
```


License & Credits
----
Gracias a [Kuangliu](https://github.com/kuangliu) por la implementación de los modelos y a [Juan Maroñas](https://github.com/jmaronas) por su ayuda en la creación de diversas funciones.

MIT - **Free Software, Hell Yeah!**


