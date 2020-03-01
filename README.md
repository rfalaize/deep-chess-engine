# Deep Chess Engine

Rest API for chess AI engine.

## Set-up

(1) Create a heroku app:

<code>heroku create <i>appname</i></code>

(2) Create a python virtual environment, and activate it:

<code>conda create --prefix <i>projectpath</i>/venv python=3.7</code>

<code>conda activate <i>projectpath</i>/venv</code>

(3) Install dependencies:

<code>pip install -r requirements.txt</code>


## Links

-   heroku app: https://deep-chess-engine.herokuapp.com/
-   heroku branch: https://git.heroku.com/deep-chess-engine.git

## Tips

Heroku settings are defined in ./Procfile

Publish local changes:

* <code>git add .</code>
* <code>git commit -m "commit msg"</code>
* <code>git push heroku master</code>

Check logs:

<code>heroku logs --tail</code>
