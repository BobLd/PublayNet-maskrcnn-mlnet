using Microsoft.ML;
using Microsoft.ML.Data;
using Microsoft.ML.Transforms.Image;
using PublayNetModelTEst.DataStructures;
using SliceAndDice;
using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Drawing;
using System.IO;
using System.Linq;
using static Microsoft.ML.Transforms.Image.ImageResizingEstimator;

namespace PublayNetModelTEst
{
    // https://github.com/PaddlePaddle/PaddleOCR/blob/0850586667308d38e113447e8d095e955092fe53/ppstructure/layout/predict_layout.py#L66
    // https://github.com/PaddlePaddle/PaddleOCR/blob/release/2.6/ppocr/postprocess/picodet_postprocess.py#L98

    class Program
    {
        private const float score_threshold = 0.4f;

        private static readonly float[] strides = new float[] { 8, 16, 32, 64 };

        static readonly Dictionary<PublayNetCategories, Color> Categories2Colors = new Dictionary<PublayNetCategories, Color>()
        {
            { PublayNetCategories.Background,   Color.AliceBlue },
            { PublayNetCategories.Text,         Color.Red },
            { PublayNetCategories.Title,        Color.Green },
            { PublayNetCategories.List,         Color.Yellow },
            { PublayNetCategories.Table,        Color.Brown },
            { PublayNetCategories.Figure,       Color.Orange }
        };

        // Unzip the model first!
        const string modelLocation = @"C:\Users\Bob\Document Layout Analysis\PaddlePaddle\picodet_lcnet_x1_0_fgd_layout_infer\onnx\picodet_lcnet_x1_0_fgd_layout_infer.onnx";

        const string imageFolder = @"Assets\Images";

        const string imageOutputFolder = @"Assets\Output";

        static void Main()
        {
            Directory.CreateDirectory(imageOutputFolder);

            MLContext mlContext = new MLContext();
            mlContext.Log += MlContext_Log;

            // Define scoring pipeline
            var pipeline = mlContext.Transforms.ResizeImages(inputColumnName: "bitmap",
                                                             outputColumnName: "image",
                                                             imageWidth: PublayNetBitmapData.NewWidth,
                                                             imageHeight: PublayNetBitmapData.NewHeight,
                                                             resizing: ResizingKind.IsoPad)
                            .Append(mlContext.Transforms.ExtractPixels(outputColumnName: "image",
                                                    scaleImage: 1f / 255f,
                                                    orderOfExtraction: ImagePixelExtractingEstimator.ColorsOrder.ARGB))
                            .Append(mlContext.Transforms.ApplyOnnxModel(inputColumnNames: new[] { "image" },
                                                                        outputColumnNames: new[]
                                                                        {
                                                                            "transpose_0.tmp_0",
                                                                            "transpose_2.tmp_0",
                                                                            "transpose_4.tmp_0",
                                                                            "transpose_6.tmp_0",

                                                                            "transpose_1.tmp_0",
                                                                            "transpose_3.tmp_0",
                                                                            "transpose_5.tmp_0",
                                                                            "transpose_7.tmp_0"
                                                                        },
                                                                        modelFile: modelLocation)); //gpuDeviceId: 0, fallbackToCpu: true,          

            // Fit on empty list to obtain input data schema
            var model = pipeline.Fit(mlContext.Data.LoadFromEnumerable(new List<PublayNetBitmapData>()));

            // Create prediction engine
            var predictionEngine = mlContext.Model.CreatePredictionEngine<PublayNetBitmapData, PublayNetPrediction>(model);

            // save model
            //mlContext.Model.Save(model, predictionEngine.OutputSchema, Path.ChangeExtension(modelLocation, "mlnet.zip"));
            //var modelLoaded = mlContext.Model.Load(Path.ChangeExtension(modelLocation, "mlnet.zip"), out var schema);

            Stopwatch stopWatch = new Stopwatch();
            foreach (var imagePath in Directory.GetFiles(imageFolder))
            {
                Console.WriteLine($"Processing {imagePath}...");

                using (var bitmap = MLImage.CreateFromFile(imagePath)) //new Bitmap(Image.FromFile(imagePath)))
                {
                    // predict
                    stopWatch.Start();

                    var img = new PublayNetBitmapData() { Image = bitmap };
                    var prediction = predictionEngine.Predict(img);

                    const int num_outs = 4;
                    var scores = new float[num_outs][] // scores or np_score_list
                    {
                        prediction.t0,
                        prediction.t2,
                        prediction.t4,
                        prediction.t6,
                    };

                    var raw_boxes = new float[num_outs][] // raw_boxes or np_boxes_list
                    {
                        prediction.t1,
                        prediction.t3,
                        prediction.t5,
                        prediction.t7,
                    };

                    /*
                        box_distribute = box_distribute[batch_id]
                        score = score[batch_id]
                        # centers
                        fm_h = input_shape[0] / stride
                        fm_w = input_shape[1] / stride
                        h_range = np.arange(fm_h)
                        w_range = np.arange(fm_w)
                        ww, hh = np.meshgrid(w_range, h_range)
                        ct_row = (hh.flatten() + 0.5) * stride
                        ct_col = (ww.flatten() + 0.5) * stride
                        center = np.stack((ct_col, ct_row, ct_col, ct_row), axis=1)
                     */

                    const int reg_max = (int)(32.0 / 4.0 - 1); // raw_boxes[0].shape[-1] = 32?

                    var a = new ArraySlice<float>(prediction.t0, new Shape(5, 7600));
                    var t = a[":, 1"];
                    t = a.GetSlice(Slice.All(), Slice.Index(1));


                    //List<float> select_scores = new List<float>();
                    //List<float[]> decode_boxes = new List<float[]>();

                    List<(float[], float[])> selected = new List<(float[], float[])>();

                    for (int i = 0; i < num_outs; i++)
                    {
                        List<(float[], float[])> selected_stride = new List<(float[], float[])>();

                        float stride = strides[i];

                        // Center
                        int fm_h = (int)(PublayNetBitmapData.NewHeight / stride);
                        int fm_w = (int)(PublayNetBitmapData.NewWidth / stride);

                        //var box_distribute = raw_boxes[i];
                        //var score = scores[i];
                        var box_distribute = new ArraySlice<float>(raw_boxes[i], new Shape(32, fm_h, fm_w)); // fm_h * fm_w));
                        var score = new ArraySlice<float>(scores[i], new Shape(5, fm_h, fm_w)); //fm_h * fm_w));

                        for (int h = 0; h < fm_h; h++)
                        {
                            for (int w = 0; w < fm_w; w++)
                            {
                                float ct_row = (h + 0.5f) * stride;
                                float ct_col = (w + 0.5f) * stride;

                                var box_distribute_c = box_distribute.GetSlice(Slice.All(), Slice.Index(h), Slice.Index(w));
                                var score_c = score.GetSlice(Slice.All(), Slice.Index(h), Slice.Index(w)).ToArray();

                                var box_distance = new ArraySlice<float>(box_distribute_c.ToArray(), new Shape(4, reg_max + 1));
                                float[] decode_box = new float[4];
                                for (int d = 0; d < 4; d++)
                                {
                                    var sm = Softmax(box_distance.GetSlice(Slice.Index(d)).ToArray());
                                    for (int k = 0; k < sm.Length; k++)
                                    {
                                        sm[k] = k * sm[k];
                                    }
                                    decode_box[d] = sm.Sum() * stride;
                                }

                                decode_box[0] = ct_col - decode_box[0];
                                decode_box[1] = ct_row - decode_box[1];
                                decode_box[2] += ct_col;
                                decode_box[3] += ct_row;

                                selected_stride.Add((score_c, decode_box));



                                /*
                                // box distribution to distance
                                for (int r = 0; r < reg_max + 1; r++) // reg_range = np.arange(reg_max + 1)
                                {

                                }
                                */

                            }
                        }

                        selected.AddRange(selected_stride.OrderByDescending(x => x.Item1.Max()).Take(1000));
                    }

                    List<(float[], float[])> picked = new List<(float[], float[])>();
                    for (int i = 0; i < selected.Count; i++)
                    {
                        var s = selected[i];
                        var bboxes = s.Item2;
                        var confidences = s.Item1;
                        var probs = confidences.Select((p, i) => new { i, p }).Where(v => v.p > score_threshold).ToArray();
                        if (probs.Length == 0) continue;

                        picked.Add(s);

                    }

                }
            }
            

            predictionEngine.Dispose();
            model.Dispose();

            Console.WriteLine("Press any key to exit.");
            Console.ReadKey();
        }


        private static float[] Softmax(float[] values)
        {
            // https://learn.microsoft.com/en-us/dotnet/machine-learning/tutorials/object-detection-onnx#create-helper-functions
            var maxVal = values.Max();
            var exp = values.Select(v => Math.Exp(v - maxVal));
            var sumExp = exp.Sum();

            return exp.Select(v => (float)(v / sumExp)).ToArray();
        }

        private static void MlContext_Log(object sender, LoggingEventArgs e)
        {
            Console.WriteLine($"{e.Kind}: {e.RawMessage}");
        }
    }
}
