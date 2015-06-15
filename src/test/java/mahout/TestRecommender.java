package mahout;

import java.io.File;
import java.io.IOException;
import java.util.List;

import junit.framework.TestCase;

import org.apache.mahout.cf.taste.common.TasteException;
import org.apache.mahout.cf.taste.impl.model.file.FileDataModel;
import org.apache.mahout.cf.taste.impl.recommender.GenericItemBasedRecommender;
import org.apache.mahout.cf.taste.impl.recommender.svd.ParallelSGDFactorizer;
import org.apache.mahout.cf.taste.impl.recommender.svd.SVDRecommender;
import org.apache.mahout.cf.taste.impl.similarity.PearsonCorrelationSimilarity;
import org.apache.mahout.cf.taste.model.DataModel;
import org.apache.mahout.cf.taste.recommender.RecommendedItem;
import org.apache.mahout.cf.taste.recommender.Recommender;
import org.apache.mahout.cf.taste.similarity.ItemSimilarity;
import org.junit.Test;

public class TestRecommender extends TestCase{
	
	private static final String FILE_PATH = "C:/tez/data/movielensDataset/ratingsComma.dat";
	private static DataModel dataModel;
	@Test
	public void testItemBased() throws TasteException, IOException {
		dataModel= getDataModel();		
        ItemSimilarity similarity = (ItemSimilarity) new PearsonCorrelationSimilarity(dataModel);
		Recommender recommender = new GenericItemBasedRecommender(dataModel, similarity);
		List<RecommendedItem> recommendedItems= recommender.recommend(1, 20);
		assertEquals(20, recommendedItems.size());
	}
	
	@Test
	public void testSVD() throws TasteException, IOException {
		dataModel= getDataModel();		
		int numFeatures=100;
		float lambda=new Float( 0.02);
		int numEpochs=20;
		ParallelSGDFactorizer factorizer=new ParallelSGDFactorizer(dataModel, numFeatures, lambda, numEpochs);
		Recommender recommender = new SVDRecommender(dataModel, factorizer);
		List<RecommendedItem> recommendedItems= recommender.recommend(1, 20);
		assertEquals(20, recommendedItems.size());
	}
	
	private FileDataModel getDataModel() throws IOException {
		return new FileDataModel (new File(FILE_PATH));
	}
}
