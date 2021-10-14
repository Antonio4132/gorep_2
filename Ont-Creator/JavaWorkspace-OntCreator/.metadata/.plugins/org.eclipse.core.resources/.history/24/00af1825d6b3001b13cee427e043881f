package single_cell;


import org.apache.jena.query.Query;
import org.apache.jena.query.QueryExecution;
import org.apache.jena.query.QueryExecutionFactory;
import org.apache.jena.query.QueryFactory;
import org.apache.jena.query.QuerySolution;
import org.apache.jena.query.ResultSet;
import org.json.JSONObject;

public class MyProject extends MyIndividual{

	private static int getSpecimenCount(MyModel model, String projectTitle) {
		
		String rdf = "http://www.w3.org/1999/02/22-rdf-syntax-ns#";
		String NS = "http://www.semanticweb.org/alicia/ontologies/2020/8/singleCellRepositories#";

		String queryString = "PREFIX a: <" + NS + "> " +
				"PREFIX rdf: <" + rdf + "> " +
				"SELECT (COUNT( * ) as ?numberOfSpecimens) \n" +
				"WHERE" +
				"{" +
					"?specimen rdf:type a:Specimen ;" +
					"          a:SPR.hasProjectTitle \"" + projectTitle + "\" ." +
				"}";
		
		
		Query query = QueryFactory.create(queryString);
		try (QueryExecution qexec = QueryExecutionFactory.create(query, model.getModel())) {
			ResultSet results = qexec.execSelect();
			
			QuerySolution soln = results.nextSolution();	
			
			return soln.getLiteral("?numberOfSpecimens").getInt();
		} catch (Exception e) {
			return 0;
		}
	}
	
	private int getCellCount(MyModel model, String projectTitle) {
		
		String rdf = "http://www.w3.org/1999/02/22-rdf-syntax-ns#";
		String NS = "http://www.semanticweb.org/alicia/ontologies/2020/8/singleCellRepositories#";

		String queryString = "PREFIX a: <" + NS + "> " +
				"PREFIX rdf: <" + rdf + "> " +
				"SELECT (SUM( ?cellCount ) as ?totalCellCount) \n" +
				"WHERE" +
				"{" +
					"?specimen rdf:type a:Specimen ;" +
					"          a:SPR.hasProjectTitle \"" + projectTitle + "\" ;" +
					"          a:SPR.hasTotalCellCount ?cellCount" +
				"}";
		
		
		Query query = QueryFactory.create(queryString);
		try (QueryExecution qexec = QueryExecutionFactory.create(query, model.getModel())) {
			ResultSet results = qexec.execSelect();
			
			QuerySolution soln = results.nextSolution();	
			
			return soln.getLiteral("?totalCellCount").getInt();
		} catch (Exception e) {
			return 0;
		}
	}
	
	public MyProject(JSONObject jsonIndividual, MyModel model) {
		super(jsonIndividual, model);
		
		String projectTitle = (String) jsonIndividual.getJSONObject("AnnotationProperties").get("SPR.hasProjectTitle");
		
		int specimenCount = getSpecimenCount(model, projectTitle);
		// int cellCount = getCellCount(model, projectTitle);
		
		jsonIndividual.getJSONObject("DataProperties").put("PR.hasSpecimenCount", specimenCount);
		// jsonIndividual.getJSONObject("DataProperties").put("SPR.hasTotalCellCount", cellCount);

	}

	@Override
	protected String[] getObjectProperties() {
		return MyModel.OBJECT_PROPERTIES;
	}

	@Override
	protected String[] getDataProperties() {
		return MyModel.PROJECT_DATA_PROPERTIES;
	}
	
	@Override
	protected String[] getAnnotationProperties() {
		return MyModel.PROJECT_ANNOTATION_PROPERTIES;
	}
	
	@Override
	protected void createIndividual(String id) {
		getModel().createProject(id);
	}


}
