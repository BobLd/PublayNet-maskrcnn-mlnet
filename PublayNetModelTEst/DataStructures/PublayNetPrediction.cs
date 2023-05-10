using Microsoft.ML.Data;
using Microsoft.ML.Transforms.Onnx;
using System;
using System.Collections.Generic;

namespace PublayNetModelTEst.DataStructures
{
    public class PublayNetPrediction
    {
        //public readonly string[] Categories2Labels = new string[] { "bg", "text", "title", "list", "table", "figure" };

        /*
         *  "transpose_0.tmp_0",
            "transpose_2.tmp_0",
            "transpose_4.tmp_0",
            "transpose_6.tmp_0",
            "transpose_1.tmp_0",
            "transpose_3.tmp_0",
            "transpose_5.tmp_0",
            "transpose_7.tmp_0"
         */

        [ColumnName("width")]
        public float ImageWidth { get; set; }

        [ColumnName("height")]
        public float ImageHeight { get; set; }

        [VectorType(7600, 5)] // float32[1,7600,5]
        [ColumnName("transpose_0.tmp_0")]
        public float[] t0 { get; set; }

        [VectorType(1900, 5)] // float32[1,1900,5]
        [ColumnName("transpose_2.tmp_0")]
        public float[] t2 { get; set; }

        [VectorType(475, 5)] // float32[1,475,5]
        [ColumnName("transpose_4.tmp_0")]
        public float[] t4 { get; set; }

        [VectorType(130, 5)] // float32[1,130,5]
        [ColumnName("transpose_6.tmp_0")]
        public float[] t6 { get; set; }

        [VectorType(7600, 32)] // float32[1,7600,32]
        [ColumnName("transpose_1.tmp_0")]
        public float[] t1 { get; set; }

        [VectorType(1900, 32)] // float32[1,1900,32]
        [ColumnName("transpose_3.tmp_0")]
        public float[] t3 { get; set; }

        [VectorType(475, 32)] // float32[1,475,32]
        [ColumnName("transpose_5.tmp_0")]
        public float[] t5 { get; set; }

        [VectorType(130, 32)] // float32[1,130,32]
        [ColumnName("transpose_7.tmp_0")]
        public float[] t7 { get; set; }

        public IEnumerable<PublayNetResult> Process(float scoreThreshold = 0.7f)
        {
            throw new NotImplementedException();
            /*
            // check outputs sizes
            if (Labels.Length != Scores.Length)
            {
                throw new ArgumentOutOfRangeException();
            }

            if (Boxes.Length != Labels.Length * 4)
            {
                throw new ArgumentOutOfRangeException();
            }

            if (Masks.Length != Labels.Length * PublayNetBitmapData.NewHeight * PublayNetBitmapData.NewWidth)
            {
                throw new ArgumentOutOfRangeException();
            }

            // compute scaling and pading
            float padx = 0;
            float padx2 = PublayNetBitmapData.NewWidth;
            float pady = 0;
            float pady2 = PublayNetBitmapData.NewHeight;

            if (ImageHeight > ImageWidth)
            {
                padx = PublayNetBitmapData.NewWidth / 2f - (ImageWidth * PublayNetBitmapData.NewWidth / ImageHeight) / 2f;
                padx2 = PublayNetBitmapData.NewWidth / 2f + (ImageWidth * PublayNetBitmapData.NewWidth / ImageHeight) / 2f;
            }
            else
            {
                pady = PublayNetBitmapData.NewHeight / 2f - (ImageHeight * PublayNetBitmapData.NewHeight / ImageWidth) / 2f;
                pady2 = PublayNetBitmapData.NewHeight / 2f + (ImageHeight * PublayNetBitmapData.NewHeight / ImageWidth) / 2f;
            }

            float scaleX = ImageWidth / (padx2 - padx);
            float scaleY = ImageHeight / (pady2 - pady);

            for (int r = 0; r < Labels.Length; r++)
            {
                var score = Scores[r];
                if (score < scoreThreshold) continue;

                // adjust bbox to page size
                var x1 = (Boxes[r * 4] - padx) * scaleX;
                var y1 = (Boxes[r * 4 + 1] - pady) * scaleY;
                var x2 = (Boxes[r * 4 + 2] - padx) * scaleX;
                var y2 = (Boxes[r * 4 + 3] - pady) * scaleY;

                // clip bbox to page bounds
                x1 = Math.Max(x1, 0);
                y1 = Math.Max(y1, 0);
                x2 = Math.Min(x2, ImageWidth - 1);
                y2 = Math.Min(y2, ImageHeight - 1);

                if (x1 > x2 || y1 > y2) continue; // remove invalid bbox

                var label = Labels[r];

                // get mask
                // TODO

                yield return new PublayNetResult(new float[] { x1, y1, x2, y2 }, score, (PublayNetCategories)label); // Categories2Labels[label]
            }
            */
        }
    }
}
