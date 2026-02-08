обновить бота на VM:

cd /opt/tg-rag-bot
git pull

# пересобрать и перезапустить
sudo docker stop tg-rag-bot
sudo docker rm tg-rag-bot
sudo docker build -t tg-rag-bot:latest .

# снова экспорт секретов (или сделаем это через systemd ниже)
export BOT_TOKEN="$(gcloud secrets versions access latest --secret=BOT_TOKEN)"
export GIGACHAT_API_KEY="$(gcloud secrets versions access latest --secret=GIGACHAT_API_KEY)"
sudo docker run -d --name tg-rag-bot \
  --restart unless-stopped \
  -e BOT_TOKEN="$BOT_TOKEN" \
  -e GIGACHAT_API_KEY="$GIGACHAT_API_KEY" \
  -e DATA_DIR=/data \
  -e LOG_LEVEL=INFO \
  -v /var/lib/tg-rag-bot:/data \
  tg-rag-bot:latest
