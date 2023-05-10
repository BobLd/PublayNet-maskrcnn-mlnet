using Microsoft.ML.Data;
using Microsoft.ML.Transforms.Image;

namespace PublayNetModelTEst.DataStructures
{
    class PublayNetBitmapData
    {
        /// <summary>
        /// Image new size after resizing step.
        /// </summary>
        public const int NewHeight = 800;
        public const int NewWidth = 608;

        [ColumnName("bitmap")]
        [ImageType(NewHeight, NewWidth)]
        public MLImage Image { get; set; }

        [ColumnName("width")]
        public float ImageWidth => Image.Width;

        [ColumnName("height")]
        public float ImageHeight => Image.Height;
    }
}
