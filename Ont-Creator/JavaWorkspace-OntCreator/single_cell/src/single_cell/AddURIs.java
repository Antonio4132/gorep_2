package single_cell;

import java.io.BufferedReader;
import java.io.FileNotFoundException;
import java.io.FileReader;
import java.io.IOException;


public class AddURIs {
	
	
	private static final String HAS_URI_RELATION = "OR.hasURI";
	
	public static void main(String[] args) {

		// Declare variables and input/output streams
		String inputFileName = "../../SingleCell-Files/singleCellRepositoriesv6.owl";
		String outputFileName = "../../SingleCell-Files/singleCellRepositoriesv6_withURIs.owl";
		String[] fileNames = new String[]{
				"../../SingleCell-Files/cell_types_ontology.csv",
				"../../SingleCell-Files/diseases_ontology.csv",
				"../../SingleCell-Files/organism_parts_ontology.csv"
				};
		
		String NS = "http://www.semanticweb.org/alicia/ontologies/2020/8/singleCellRepositories#";

		MyModel model = new MyModel(NS, inputFileName);

		for (String fileName : fileNames) {
			// Read csv file with cell_types and loop over cell types
			try (BufferedReader br = new BufferedReader(new FileReader(fileName))) {
			    String line;
			    
			    while ((line = br.readLine()) != null) {
			        String[] values = line.split("\t");
			        
			        try {
				        String name = values[0];
				        String URI = values[1];
				        
						// Add URI to each cell type
				        model.addAnotationPropertyToIndividual(name, HAS_URI_RELATION, URI);
					} catch (Exception e) {
					}

			    }
			    
			} catch (FileNotFoundException e) {
				// TODO Auto-generated catch block
				e.printStackTrace();
			} catch (IOException e) {
				// TODO Auto-generated catch block
				e.printStackTrace();
			}
		}
		

		// Check if model is valid
		if (!model.validateModel())
			;

		// Save the model with the instances
		model.saveModel(outputFileName);

	}
}
