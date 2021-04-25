# detectron2_web_app

https://towardsdatascience.com/instance-segmentation-web-app-63016b8ed4ae

How to build locally:
```
gsutil cp gs://sbir-ii-models/detectron2/exp-1/model_0088499.pth model.pth
docker build --build-arg model_version=<model version> --build-arg model_iteration=<model iteration> -t detectron2-serve .
docker tag detectron2-serve:latest gcr.io/sbir-training/detectron2-serve:latest
docker push gcr.io/sbir-training/detectron2-serve:latest
docker run --rm -it -p 8080:8080 detectron2-serve
```

How to build and deploy:
```
gsutil cp gs://sbir-ii-models/detectron2/exp-1/model_0088499.pth model.pth
gcloud builds submit --project "sbir-training" --tag gcr.io/sbir-training/detectron2-serve --timeout=86400 && \
gcloud run deploy detectron2-serve --project "sbir-training" --image gcr.io/sbir-training/detectron2-serve:latest --region "us-west1" --platform managed --memory 2G --allow-unauthenticated
```


