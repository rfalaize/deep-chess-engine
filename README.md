# Deep Chess Engine

Rest API for chess AI engine.

## Set-up

Create a heroku app and install redis addon:

-   heroku create <appname>
-   heroku addons:create heroku-redis -a sushi

## Links

-   heroku app: https://deep-chess-engine.herokuapp.com/
-   heroku branch: https://git.heroku.com/deep-chess-engine.git

## Tips

Heroku settings are defined in ./Procfile

Publish local changes:

<code>git add .</code>
<code>git commit -m "commit msg"</code>
<code>git push heroku master</code>

Check logs:

<code>heroku logs --tail</code>
