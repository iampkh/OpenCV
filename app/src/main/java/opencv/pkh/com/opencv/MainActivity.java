package opencv.pkh.com.opencv;

import android.graphics.Bitmap;
import android.graphics.BitmapFactory;
import android.support.v7.app.AppCompatActivity;
import android.os.Bundle;
import android.util.DisplayMetrics;
import android.util.Log;
import android.view.SurfaceView;
import android.widget.Toast;

import org.opencv.android.BaseLoaderCallback;
import org.opencv.android.CameraBridgeViewBase;
import org.opencv.android.JavaCameraView;
import org.opencv.android.OpenCVLoader;
import org.opencv.android.Utils;
import org.opencv.core.Core;
import org.opencv.core.CvType;
import org.opencv.core.Mat;
import org.opencv.core.MatOfPoint;
import org.opencv.core.MatOfPoint2f;
import org.opencv.core.MatOfRect;
import org.opencv.core.Point;
import org.opencv.core.Rect;
import org.opencv.core.Scalar;
import org.opencv.core.Size;
import org.opencv.imgcodecs.Imgcodecs;
import org.opencv.imgproc.Imgproc;
import org.opencv.objdetect.CascadeClassifier;

import java.io.File;
import java.io.FileOutputStream;
import java.io.InputStream;
import java.util.ArrayList;
import java.util.List;

public class MainActivity extends AppCompatActivity implements CameraBridgeViewBase.CvCameraViewListener2{

    CameraBridgeViewBase mCameraBridgeViewBase;
    Mat mGrayScaleMat,mRgba,mRgbaF,mRgbaT,mIntermediate,mHeirarchy,mMastClone;

    BaseLoaderCallback mBaseLoaderCallback;

    CascadeClassifier mCascadeClassifier,mEyeCascadeClassifer;

    int mAbsoluteLogoSize;
    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);

        Toast.makeText(this, " OpenCv Sucess", Toast.LENGTH_SHORT).show();
        mCameraBridgeViewBase = (JavaCameraView) findViewById(R.id.cameraView);
        mCameraBridgeViewBase.setVisibility(SurfaceView.VISIBLE);
        mCameraBridgeViewBase.setCameraIndex(1);
        mCameraBridgeViewBase.setCvCameraViewListener(this);

        mBaseLoaderCallback = new BaseLoaderCallback(this) {
            @Override
            public void onManagerConnected(int status) {

                switch (status) {
                    case BaseLoaderCallback.SUCCESS:
                        mCameraBridgeViewBase.enableView();
                        initOpenCvDependency();
                        break;
                    default:
                        super.onManagerConnected(status);
                        break;
                }
            }
        };

    }

    private void initOpenCvDependency() {

        try {
            //Read the cascade clasifier xml from raw folder
            InputStream isFace = getResources().openRawResource(R.raw.cascade);
            InputStream isEye = getResources().openRawResource(R.raw.haarcascade_eye);

            File cascadeDire = getDir("cascade", MODE_PRIVATE);
            File eyecascadeDire = getDir("eyecascade", MODE_PRIVATE);

            File cascadeFile = new File(cascadeDire, "cascade.xml");
            File eyeCascadeFile = new File(eyecascadeDire,"haarcascade_eye.xml");

            FileOutputStream osFace = new FileOutputStream(cascadeFile);
            FileOutputStream osEye = new FileOutputStream(eyeCascadeFile);

            byte[] buffer = new byte[4096];
            int byteRead;
            while ((byteRead = isFace.read(buffer)) != -1) {
                osFace.write(buffer,0,byteRead);
            }
            buffer = new byte[4096];
            while ((byteRead = isEye.read(buffer)) != -1) {
                osEye.write(buffer,0,byteRead);
            }
            isFace.close();
            isEye.close();

            osFace.close();
            osEye.close();

            Log.d("pkhfacedetect" , "cascade path ="+cascadeFile.getAbsolutePath());
            mCascadeClassifier = new CascadeClassifier(cascadeFile.getAbsolutePath());
            mEyeCascadeClassifer = new CascadeClassifier(eyeCascadeFile.getAbsolutePath());


        }catch (Exception e) {
            Log.d("pkhfacedetect" , "Errorrr loading cascade classifier"+e.toString());

            e.printStackTrace();
        }
        mCameraBridgeViewBase.enableView();
    }

    @Override
    public void onCameraViewStarted(int width, int height) {

        mGrayScaleMat = new Mat(height, width, CvType.CV_8UC4);

        mAbsoluteLogoSize = (int)(height* 0.2);

        mRgba = new Mat(height, width, CvType.CV_8UC4);
        mRgbaF = new Mat(height, width, CvType.CV_8UC4);
        mRgbaT = new Mat(width, width, CvType.CV_8UC4);
        mIntermediate = new Mat(width, width, CvType.CV_8UC4);

    }

    @Override
    public void onCameraViewStopped() {
        mGrayScaleMat.release();

    }

    public Mat detectFace(Mat mat) {

        //read bitmap
        Bitmap tonyBmp = BitmapFactory.decodeResource(this.getResources(),R.drawable.tony);
        Mat tonyMat = new Mat();
        Utils.bitmapToMat(tonyBmp,tonyMat);
        int[] screenSize =  getScreenSize();
        Size matSize = new Size(screenSize[0],screenSize[1]);
        Imgproc.resize(tonyMat,tonyMat,matSize);

        tonyMat = mat;
        MatOfRect matOfRectFaceVectors = new MatOfRect();
        MatOfRect eyeDetections = new MatOfRect();

        mCascadeClassifier.detectMultiScale(tonyMat,matOfRectFaceVectors);

        Log.d("pkhfacedetect" , matOfRectFaceVectors.toArray().length + " faces found");

        for (Rect rect : matOfRectFaceVectors.toArray()) {
            /*Imgproc.rectangle(tonyMat, new Point(rect.x, rect.y), new Point(rect.x
                    + rect.width, rect.y + rect.height), new Scalar(0, 255, 0));*/


            //Kare çiz
            Imgproc.rectangle(tonyMat, new Point(rect.x, rect.y), new Point(rect.x + rect.width, rect.y + rect.height),
                    new Scalar(200, 200, 100),2);

        }
        Mat face;
        Mat crop = null;
        Mat circles = new Mat();
        for(int i=0;i<matOfRectFaceVectors.toArray().length;i++){

            face = tonyMat.submat(matOfRectFaceVectors.toArray()[i]);
            crop = face.submat(4, (2*face.width())/3, 0, face.height());

            mEyeCascadeClassifer.detectMultiScale(crop, eyeDetections);//, 1.1, 2, 0,new Size(30,30), new Size());

            for (int j = 0; j< eyeDetections.toArray().length ; j++){

                System.out.println("Eye" );

                Mat eye = crop.submat(eyeDetections.toArray()[j]);

                Rect rect = eyeDetections.toArray()[j];
                Imgproc.putText(tonyMat, "Eye", new Point(rect.x,rect.y-5), 1, 2, new Scalar(0,0,255));
                Imgproc.rectangle(tonyMat, new Point(rect.x, rect.y), new Point(rect.x + rect.width, rect.y + rect.height),
                        new Scalar(200, 200, 100),2);
                /*eyes = "Eye"+j+".png";
                Imgcodecs.imwrite(eyes, eye);*/

            }
        }

        return tonyMat;

    }

    public Mat detectFace2(Mat mat) {

        //read bitmap
        Bitmap tonyBmp = BitmapFactory.decodeResource(this.getResources(),R.drawable.tony);
        Mat tonyMat = new Mat();
        Utils.bitmapToMat(tonyBmp,tonyMat);
        int[] screenSize =  getScreenSize();
        Size matSize = new Size(screenSize[0],screenSize[1]);
        Imgproc.resize(tonyMat,tonyMat,matSize);

        tonyMat = mat;
        MatOfRect matOfRectFaceVectors = new MatOfRect();
        MatOfRect eyeDetections = new MatOfRect();

        mCascadeClassifier.detectMultiScale(tonyMat,matOfRectFaceVectors);

        Log.d("pkhfacedetect" , matOfRectFaceVectors.toArray().length + " faces found");

        for (Rect rect : matOfRectFaceVectors.toArray()) {
            /*Imgproc.rectangle(tonyMat, new Point(rect.x, rect.y), new Point(rect.x
                    + rect.width, rect.y + rect.height), new Scalar(0, 255, 0));*/


            //Kare çiz
            Imgproc.rectangle(tonyMat, new Point(rect.x, rect.y), new Point(rect.x + rect.width, rect.y + rect.height),
                    new Scalar(200, 200, 100),2);

        }
        Mat face;
        Mat crop = null;
        Mat circles = new Mat();
        for(int i=0;i<matOfRectFaceVectors.toArray().length;i++){

            face = tonyMat.submat(matOfRectFaceVectors.toArray()[i]);
            crop = face.submat(4, (2*face.width())/3, 0, face.height());

            mEyeCascadeClassifer.detectMultiScale(crop, eyeDetections, 1.1, 2, 0,new Size(30,30), new Size());

            for (int j = 0; j< eyeDetections.toArray().length ; j++){

                System.out.println("Eye" );

                Mat eye = crop.submat(eyeDetections.toArray()[j]);

                Rect rect = eyeDetections.toArray()[j];
                Imgproc.putText(tonyMat, "Eye", new Point(rect.x,rect.y-5), 1, 2, new Scalar(0,0,255));
                Imgproc.rectangle(tonyMat, new Point(rect.x, rect.y), new Point(rect.x + rect.width, rect.y + rect.height),
                        new Scalar(200, 200, 100),2);
                /*eyes = "Eye"+j+".png";
                Imgcodecs.imwrite(eyes, eye);*/

            }
        }

        return tonyMat;

    }

    public int[] getScreenSize(){
        DisplayMetrics displayMetrics = new DisplayMetrics();
        getWindowManager().getDefaultDisplay().getMetrics(displayMetrics);
        int height = displayMetrics.heightPixels;
        int width = displayMetrics.widthPixels;

        int[] size = new int[2];
        size[0] = height;
        size[1] = width;

        return size;
    }

    @Override
    public Mat onCameraFrame(CameraBridgeViewBase.CvCameraViewFrame inputFrame) {
       // detectFace(null);
        Imgproc.cvtColor(inputFrame.rgba(), mGrayScaleMat, Imgproc.COLOR_RGBA2RGB);

        mRgba = inputFrame.rgba();
        // Rotate mRgba 90 degrees
        Core.transpose(mRgba, mRgbaT);
       // Core.rotate(mRgbaT,mRgba,Core.ROTATE_90_COUNTERCLOCKWISE);
        Imgproc.resize(mRgbaT, mRgbaF, mRgbaF.size(), 0,0, 0);
        Core.flip(mRgbaF, mRgba, Core.ROTATE_90_COUNTERCLOCKWISE);
        //Core.
        mRgba = detectFace2(mRgba);

        Log.d("pkhDebug", "onCameraFragme ....");
        return mRgba;
    }

    @Override
    protected void onPause() {
        super.onPause();
        if (mCameraBridgeViewBase != null){
            mCameraBridgeViewBase.disableView();
        }
    }

    @Override
    protected void onResume() {
        super.onResume();

        if (!OpenCVLoader.initDebug()){
            Toast.makeText(this,"OpenCV Failure", Toast.LENGTH_SHORT);
            OpenCVLoader.initAsync(OpenCVLoader.OPENCV_VERSION_2_4_11,
                    this, mBaseLoaderCallback);
        }else {
            mBaseLoaderCallback.onManagerConnected(BaseLoaderCallback.SUCCESS);
        }
    }

    @Override
    protected void onDestroy() {
        super.onDestroy();
        if (mCameraBridgeViewBase != null){
            mCameraBridgeViewBase.disableView();
        }
    }
}
