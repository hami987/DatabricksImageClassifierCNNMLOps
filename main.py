import mlflow
from models.cnn import build_model, SklearnModelWrapper
from data.preprocessing import download_preprocess
from data.plot_data import plot_10

if __name__ == "__main__":
    
    model = build_model()
    model = SklearnModelWrapper(model=model)
    train_images, train_labels, test_images, test_labels, class_names = download_preprocess()
    plot_10(train_images=train_images, class_names=class_names, train_labels=train_labels)
    
    with mlflow.start_run(run_name='cnn_image_classifier_demo'):
        cnn=model.compile(optimizer='adam', 
                    loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                    metrics=['accuracy'])
        history = model.fit(train_images, train_labels, epochs=10, 
                            validation_data=(test_images, test_labels))
        mlflow.log_metric('accuracy', history.history['accuracy'][-1])
        wrappedModel = SklearnModelWrapper(model)
        mlflow.pyfunc.log_model("cnn_image_classifier_demo", python_model=wrappedModel)