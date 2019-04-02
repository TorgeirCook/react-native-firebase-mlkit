
package com.mlkit;

import android.app.Activity;
import android.content.Context;
import android.graphics.Rect;
import android.hardware.camera2.CameraAccessException;
import android.hardware.camera2.CameraCharacteristics;
import android.hardware.camera2.CameraManager;
import android.net.Uri;
import android.os.Build;
import android.support.annotation.NonNull;
import android.util.Log;
import android.util.SparseIntArray;
import android.view.Surface;

import com.facebook.react.bridge.Arguments;
import com.facebook.react.bridge.Promise;
import com.facebook.react.bridge.ReactApplicationContext;
import com.facebook.react.bridge.ReactContextBaseJavaModule;
import com.facebook.react.bridge.ReactMethod;
import com.facebook.react.bridge.WritableArray;
import com.facebook.react.bridge.WritableMap;
import com.google.android.gms.tasks.OnFailureListener;
import com.google.android.gms.tasks.OnSuccessListener;
import com.google.android.gms.tasks.Task;
import com.google.firebase.ml.vision.FirebaseVision;
import com.google.firebase.ml.vision.common.FirebaseVisionImage;
import com.google.firebase.ml.vision.common.FirebaseVisionImageMetadata;
import com.google.firebase.ml.vision.document.FirebaseVisionDocumentText;
import com.google.firebase.ml.vision.document.FirebaseVisionDocumentTextRecognizer;
import com.google.firebase.ml.vision.text.FirebaseVisionText;
import com.google.firebase.ml.vision.text.FirebaseVisionTextRecognizer;

import java.io.ByteArrayOutputStream;
import java.io.IOException;
import java.io.InputStream;
import java.nio.ByteBuffer;
import java.util.List;

import androidx.annotation.RequiresApi;

import static android.content.Context.CAMERA_SERVICE;
import static android.hardware.Camera.*;

public class RNMlKitModule extends ReactContextBaseJavaModule {

    private static final SparseIntArray ORIENTATIONS = new SparseIntArray();
    private static final String TAG = "RNMlKit";

    static {
        ORIENTATIONS.append(Surface.ROTATION_0, 90);
        ORIENTATIONS.append(Surface.ROTATION_90, 0);
        ORIENTATIONS.append(Surface.ROTATION_180, 270);
        ORIENTATIONS.append(Surface.ROTATION_270, 180);
    }

    private final ReactApplicationContext reactContext;
    private FirebaseVisionTextRecognizer textDetector;
    private FirebaseVisionTextRecognizer cloudTextDetector;
    private FirebaseVisionDocumentTextRecognizer cloudDocumentDetector;

    public RNMlKitModule(ReactApplicationContext reactContext) {
        super(reactContext);
        this.reactContext = reactContext;
    }

    @ReactMethod
    public void deviceTextRecognition(String uri, final Promise promise) {
        try {
            FirebaseVisionImage image = FirebaseVisionImage.fromFilePath(this.reactContext, android.net.Uri.parse(uri));
            FirebaseVisionTextRecognizer detector = this.getTextRecognizerInstance();
            Task<FirebaseVisionText> result =
                    detector.processImage(image)
                            .addOnSuccessListener(new OnSuccessListener<FirebaseVisionText>() {
                                @Override
                                public void onSuccess(FirebaseVisionText firebaseVisionText) {
                                    promise.resolve(processDeviceResult(firebaseVisionText));
                                }
                            })
                            .addOnFailureListener(
                                    new OnFailureListener() {
                                        @Override
                                        public void onFailure(@NonNull Exception e) {
                                            e.printStackTrace();
                                            promise.reject(e);
                                        }
                                    });
            ;
        } catch (IOException e) {
            promise.reject(e);
            e.printStackTrace();
        }
    }

    private FirebaseVisionTextRecognizer getTextRecognizerInstance() {
        if (this.textDetector == null) {
            this.textDetector = FirebaseVision.getInstance().getOnDeviceTextRecognizer();
        }

        return this.textDetector;
    }

    @ReactMethod
    public void close(final Promise promise) {
        if (this.textDetector != null) {
            try {
                this.textDetector.close();
                this.textDetector = null;
                promise.resolve(true);
            } catch (IOException e) {
                e.printStackTrace();
                promise.reject(e);
            }
        }

        if (this.cloudTextDetector != null) {
            try {
                this.cloudTextDetector.close();
                this.cloudTextDetector = null;
                promise.resolve(true);
            } catch (IOException e) {
                e.printStackTrace();
                promise.reject(e);
            }
        }
    }

    private FirebaseVisionTextRecognizer getCloudTextRecognizerInstance() {
        if (this.cloudTextDetector == null) {
            this.cloudTextDetector = FirebaseVision.getInstance().getCloudTextRecognizer();
        }

        return this.cloudTextDetector;
    }

    private FirebaseVisionDocumentTextRecognizer getCloudTextDocumentRecognizerInstance() {
        if (this.cloudDocumentDetector == null) {
            this.cloudDocumentDetector = FirebaseVision.getInstance().getCloudDocumentTextRecognizer();
        }

        return this.cloudDocumentDetector;
    }

    @RequiresApi(api = Build.VERSION_CODES.LOLLIPOP)
    @ReactMethod
    public void cloudTextRecognition(String uri, final Promise promise) throws CameraAccessException {
        try {
            int rotation = getRotationCompensation(String.valueOf(getCameraId()), getCurrentActivity(), reactContext.getBaseContext());
            FirebaseVisionImageMetadata metadata = new FirebaseVisionImageMetadata.Builder()
                    .setWidth(480)   // 480x360 is typically sufficient for
                    .setHeight(360)  // image recognition
                    .setFormat(FirebaseVisionImageMetadata.IMAGE_FORMAT_NV21)
                    .setRotation(rotation)
                    .build();
            FirebaseVisionDocumentTextRecognizer detector = this.getCloudTextDocumentRecognizerInstance();
            byte[] bytes = getBytes(uri);
            FirebaseVisionImage image = FirebaseVisionImage.fromByteBuffer(ByteBuffer.wrap(bytes), metadata);
//            FirebaseVisionImage image = FirebaseVisionImage.fromFilePath(this.reactContext, android.net.Uri.parse(uri));
            Task<FirebaseVisionDocumentText> result =
                    detector.processImage(image)
                            .addOnSuccessListener(new OnSuccessListener<FirebaseVisionDocumentText>() {
                                @Override
                                public void onSuccess(FirebaseVisionDocumentText firebaseVisionDocumentText) {
                                    promise.resolve(processCloudResult(firebaseVisionDocumentText));
                                }
                            })
                            .addOnFailureListener(
                                    new OnFailureListener() {
                                        @Override
                                        public void onFailure(@NonNull Exception e) {
                                            e.printStackTrace();
                                            promise.reject(e);
                                        }
                                    });
        } catch (IOException e) {
            promise.reject(e);
            e.printStackTrace();
        }
    }

    private WritableArray processDeviceResult(FirebaseVisionText firebaseVisionText) {
        WritableArray data = Arguments.createArray();
        WritableMap info;
        WritableMap coordinates;
        List<FirebaseVisionText.TextBlock> blocks = firebaseVisionText.getTextBlocks();

        if (blocks.size() == 0) {
            return data;
        }

        for (int i = 0; i < blocks.size(); i++) {
            List<FirebaseVisionText.Line> lines = blocks.get(i).getLines();
            info = Arguments.createMap();
            coordinates = Arguments.createMap();

            Rect boundingBox = blocks.get(i).getBoundingBox();
            coordinates.putInt("top", boundingBox.top);
            coordinates.putInt("left", boundingBox.left);
            coordinates.putInt("width", boundingBox.width());
            coordinates.putInt("height", boundingBox.height());

            info.putMap("blockCoordinates", coordinates);
            info.putString("blockText", blocks.get(i).getText());
            info.putString("resultText", firebaseVisionText.getText());

            for (int j = 0; j < lines.size(); j++) {
                List<FirebaseVisionText.Element> elements = lines.get(j).getElements();
                info.putString("lineText", lines.get(j).getText());

                for (int k = 0; k < elements.size(); k++) {
                    info.putString("elementText", elements.get(k).getText());
                }
            }

            data.pushMap(info);
        }

        return data;
    }

    private WritableArray processCloudResult(FirebaseVisionDocumentText documentText) {
        List<FirebaseVisionDocumentText.Block> blocks = documentText.getBlocks();


        WritableArray blocksArr = Arguments.createArray();
        if (blocks.size() == 0) {
            return blocksArr;
        }

        for (FirebaseVisionDocumentText.Block block : blocks) {
            String blockText = block.getText();
            Float blockConfidence = block.getConfidence();
            Rect blockFrame = block.getBoundingBox();
            WritableMap blockMap = createMap(blockText, blockConfidence, blockFrame);
            WritableArray paragraphArr = Arguments.createArray();

            for (FirebaseVisionDocumentText.Paragraph paragraph : block.getParagraphs()) {
                String paragraphText = paragraph.getText();
                Float paragraphConfidence = paragraph.getConfidence();
                Rect paragraphFrame = paragraph.getBoundingBox();
                WritableMap paragraphMap = createMap(paragraphText, paragraphConfidence, paragraphFrame);
//                WritableArray wordsArr = Arguments.createArray();

//                for (FirebaseVisionDocumentText.Word word : paragraph.getWords()) {
//                    String wordText = word.getText();
//                    Float wordConfidence = word.getConfidence();
//                    Rect wordFrame = word.getBoundingBox();
//                    WritableMap wordMap = createMap(wordText, wordConfidence, wordFrame);
//                    WritableArray symbolsArr = Arguments.createArray();
//
//                    for (FirebaseVisionDocumentText.Symbol symbol : word.getSymbols()) {
//                        String symbolText = symbol.getText();
//                        Float symbolConfidence = symbol.getConfidence();
//                        Rect symbolFrame = symbol.getBoundingBox();
//                        WritableMap symbolMap = createMap(symbolText, symbolConfidence, symbolFrame);
//                        symbolsArr.pushMap(symbolMap);
//                    }
//                    wordMap.putArray("symbols", symbolsArr);
//                    wordsArr.pushMap(wordMap);
//                }
//                paragraphMap.putArray("words", wordsArr);
                paragraphArr.pushMap(paragraphMap);

            }
            blockMap.putArray("paragraphs", paragraphArr);
            blocksArr.pushMap(blockMap);
        }
        return blocksArr;
    }

    private WritableMap createMap(String text, Float confidence, Rect frame) {
        WritableMap map = Arguments.createMap();
        WritableMap rectMap = Arguments.createMap();
        map.putString("text", text);
        map.putDouble("confidence", confidence);
        rectMap.putInt("bottom", frame.bottom);
        rectMap.putInt("right", frame.right);
        rectMap.putInt("left", frame.left);
        rectMap.putInt("top", frame.top);
        rectMap.putInt("width", frame.width());
        rectMap.putInt("height", frame.height());
        map.putMap("coordinates", rectMap);
        return map;
    }

    /**
     * Get the angle by which an image must be rotated given the device's current
     * orientation.
     */
    @RequiresApi(api = Build.VERSION_CODES.LOLLIPOP)
    private int getRotationCompensation(String cameraId, Activity activity, Context context)
            throws CameraAccessException {
        // Get the device's current rotation relative to its "native" orientation.
        // Then, from the ORIENTATIONS table, look up the angle the image must be
        // rotated to compensate for the device's rotation.
        int deviceRotation = activity.getWindowManager().getDefaultDisplay().getRotation();
        int rotationCompensation = ORIENTATIONS.get(deviceRotation);

        // On most devices, the sensor orientation is 90 degrees, but for some
        // devices it is 270 degrees. For devices with a sensor orientation of
        // 270, rotate the image an additional 180 ((270 + 270) % 360) degrees.
        CameraManager cameraManager = (CameraManager) context.getSystemService(CAMERA_SERVICE);
        int sensorOrientation = cameraManager
                .getCameraCharacteristics(cameraId)
                .get(CameraCharacteristics.SENSOR_ORIENTATION);
        rotationCompensation = (rotationCompensation + sensorOrientation + 270) % 360;

        // Return the corresponding FirebaseVisionImageMetadata rotation value.
        int result;
        switch (rotationCompensation) {
            case 0:
                result = FirebaseVisionImageMetadata.ROTATION_0;
                break;
            case 90:
                result = FirebaseVisionImageMetadata.ROTATION_90;
                break;
            case 180:
                result = FirebaseVisionImageMetadata.ROTATION_180;
                break;
            case 270:
                result = FirebaseVisionImageMetadata.ROTATION_270;
                break;
            default:
                result = FirebaseVisionImageMetadata.ROTATION_0;
                Log.e(TAG, "Bad rotation value: " + rotationCompensation);
        }
        return result;
    }

    private byte[] getBytes(String uri) throws IOException {
        InputStream inputStream = this.reactContext.getContentResolver().openInputStream(Uri.parse(uri));
        ByteArrayOutputStream byteBuffer = new ByteArrayOutputStream();
        int bufferSize = 1024;
        byte[] buffer = new byte[bufferSize];

        int len = 0;
        while ((len = inputStream.read(buffer)) != -1) {
            byteBuffer.write(buffer, 0, len);
        }
        return byteBuffer.toByteArray();
    }

    private int getCameraId() {
        int cameraId = -1;
        int numberOfCameras = getNumberOfCameras();
        for (int i = 0; i < numberOfCameras; i++) {
            CameraInfo info = new CameraInfo();
            getCameraInfo(i, info);
            if (info.facing == CameraInfo.CAMERA_FACING_BACK) {
                Log.d(TAG, "Camera found");
                cameraId = i;
                break;
            }
        }
        return cameraId;
    }


    @Override
    public String getName() {
        return "RNMlKit";
    }
}
