package single_cell;

import java.io.IOException;

import org.json.*;

import java.nio.file.Files;
import java.nio.file.Path;

public class Test {

	private static String readJSON2String(String fileName) {
		Path fileNamePath = Path.of(fileName);
         
        String jsonContent = null;
		try {
			jsonContent = Files.readString(fileNamePath);
		} catch (IOException e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		}
        
		return jsonContent;
	}
	
	
	public static void main(String[] args) {
		
		// Declare variables and input/output streams
		String inputFileName = "../../../SingleCell-Files/singleCellRepositoriesv6_withURIs.owl";
		String outputFileName = "../../../SingleCell-Files/out_repositoriev6.owl";
		String[] hitsFileNames = new String[]{
				"../../../SingleCell-Files/processed_data/HCA_processed.json",
				"../../../SingleCell-Files/processed_data/SCAE_processed.json"
				};
		
		String NS = "http://www.semanticweb.org/alicia/ontologies/2020/8/singleCellRepositories#";

		MyModel model = new MyModel(NS, inputFileName);

		for (String fileName : hitsFileNames) {
			System.out.println("Adding to model " + fileName);
			// Read samples to introduce into the model as instances
			String rawJson = readJSON2String(fileName);
			
			// Parse JSON to obtain the specimens and the projects
			JSONObject fullJson = new JSONObject(rawJson);
			JSONArray specimensArray = fullJson.getJSONArray("specimens");
			JSONArray projectsArray = fullJson.getJSONArray("projects");
					
			System.out.println("Adding specimens to model...");
			specimensArray.forEach((specimen) -> {
				MySpecimen mySpecimen = new MySpecimen((JSONObject) specimen, model);
			//	System.out.println(mySpecimen.getId());
				mySpecimen.addToModel();
			});
			
			System.out.println("Adding projects to model...");
			projectsArray.forEach((project) -> {
				MyProject myProject = new MyProject((JSONObject) project, model);
			//	System.out.println(myProject.getId());
				myProject.addToModel();
			});
		}

		// Check if model is valid
		if (!model.validateModel())
			;
		
		// Save the model with the instances
		model.saveModel(outputFileName);

	}

}


// TODO Add instances with subject-predicate-object structure
//if (flag) {
//	Resource subject = ResourceFactory.createResource("dummy/dummy");
//	Property predicate = RDF.type;
//	Resource object = ResourceFactory.createResource(NS + "HomoSapiens");
//	Statement statement = ResourceFactory.createStatement(subject, predicate, object);
//
//	model.add(statement);
//}
