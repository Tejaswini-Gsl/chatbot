gcloud run deploy info-chatbot \
  --source . \
  --set-env-vars OPENAI_API_KEY=KEY \
  --region us-central1 \
  --allow-unauthenticated \
  --max-instances 2 \
  --min-instances 1 \
  


