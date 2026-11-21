FROM python:3.13-slim
WORKDIR /bot
COPY mechfight.py nav.py bot.py /bot/
ENTRYPOINT ["python3", "-u", "/bot/bot.py"]
