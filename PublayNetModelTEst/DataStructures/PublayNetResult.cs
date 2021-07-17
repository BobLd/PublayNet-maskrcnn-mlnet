namespace PublayNetModelTEst.DataStructures
{
    public class PublayNetResult
    {
        internal PublayNetResult(float[] bbox, float score, PublayNetCategories category)
        {
            Bbox = bbox;
            Score = score;
            Category = category;
        }

        /// <summary>
        /// x1, y1, x2, y2.
        /// </summary>
        public float[] Bbox { get; }

        public float Score { get;}

        public PublayNetCategories Category { get; }
    }
}
