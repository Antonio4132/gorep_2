package single_cell;

import org.apache.jena.query.Query;
import org.apache.jena.query.QueryExecution;
import org.apache.jena.query.QueryExecutionFactory;
import org.apache.jena.query.QueryFactory;
import org.apache.jena.query.QuerySolution;
import org.apache.jena.query.ResultSet;

public class QueryAlicia {
	
	private static void executeQuery(String NS, MyModel model, String queryStringTest7) {
		Query query = QueryFactory.create(queryStringTest7);
		try (QueryExecution qexec = QueryExecutionFactory.create(query, model.getModel())) {
			ResultSet results = qexec.execSelect();
			int i = 0;
			for (; results.hasNext();i++) {
				QuerySolution soln = results.nextSolution();
				
				System.out.println(soln.toString().replaceAll(NS, ""));
			}
			// System.out.println(i + " results.");
		}
	}
	
	public static void main(String[] args) {
		String inputFileName = "../../SingleCell-Files/out_repositoriev4.owl";

		String NS = "http://www.semanticweb.org/alicia/ontologies/2020/8/singleCellRepositories#";
		String rdf = "http://www.w3.org/1999/02/22-rdf-syntax-ns#";
		String xsd = "http://www.w3.org/2001/XMLSchema#";
		
		MyModel model = new MyModel(NS, inputFileName);
		
		/* 1. COMPROBACIÓN
		 * En primer lugar, he pensado que podríamos hacer algunas consultas que nos permitan comprobar que hemos incluido 
		 * toda la información de la HCA, contando el número de proyectos, laboratorios, especímenes, recuento de células y tamaño 
		 * total de los ficheros en GB (también hay otra que incluye el número de ficheros, pero nosotros no lo hemos incluido, no 
		 * creo que sea relevante). También podríamos hacerlo para órganos aunque ya sepamos que no va a coincidir para tener un resumen
		 *  completo de las características de nuestro dataset!  (si vemos que hay mucha discrepancia, lo ideal sería  repetir la misma 
		 *  consulta para cada uno de los órganos del HCA).
		 *  
		 *  Una segunda propuesta sería, por ejemplo, sumar el tipo de células para los principales objetos de estudio: blood, kidney, 
		 *  bone, liver, brain, lung, pancreas, heart, immune system y skin.
		 */
		
		// -----------------------------
		// 1.1. Número de células
		// -----------------------------
		System.out.println();
		System.out.println("------------------------");
		System.out.println("1.1. Número de células");
		System.out.println("------------------------");
		System.out.println();
		
		String queryStringNumCells = "PREFIX a: <" + NS + "> " +
				"PREFIX rdf: <" + rdf + "> " +
				"PREFIX xsd: <" + xsd + "> " +
				"SELECT (SUM(?numCells)/1000000 as ?numberOfCellsProject) \n" +
				"WHERE" +
				"{" +
					"?id rdf:type a:Project ;" +
					"    a:SPR.hasTotalCellCount ?numCells ." +
					"FILTER (?numCells != -1)" +
				"}";
		
		// Execute query
		executeQuery(NS, model, queryStringNumCells);
		
		// -----------------------------
		// 1.2. Número de órganos
		// -----------------------------
		System.out.println();
		System.out.println("------------------------");
		System.out.println("1.2. Número de órganos");
		System.out.println("------------------------");
		System.out.println();
		
		String queryStringNumOrgans = "PREFIX a: <" + NS + "> " +
				"PREFIX rdf: <" + rdf + "> " +
				"PREFIX xsd: <" + xsd + "> " +
				"SELECT (COUNT( DISTINCT ?organismPart ) as ?numberOfOrgansProject) \n" +
				"WHERE" +
				"{" +
					"?id rdf:type a:Project ;" +
					"    a:SPR.hasOrganismPart ?organismPart ." +
					"?organismPart rdf:type a:Organ ." +
				"}";
		
		// Execute query
		executeQuery(NS, model, queryStringNumOrgans);
		
		// -----------------------------
		// 1.3. Número de donantes
		// -----------------------------
		System.out.println();
		System.out.println("------------------------");
		System.out.println("1.3. Número de donantes");
		System.out.println("------------------------");
		System.out.println();
		
		String queryStringDonors = "PREFIX a: <" + NS + "> " +
				"PREFIX rdf: <" + rdf + "> " +
				"PREFIX xsd: <" + xsd + "> " +
				"SELECT ( SUM(?donorCount) AS ?totalDonorCount ) \n" +
				"WHERE" +
				"{" +
					"?id rdf:type a:Project ;" +
					"    a:PR.hasDonorCount ?donorCount ." +
					"FILTER (?donorCount != -1) " +
				"}" +
				"ORDER BY ?project";
		
		// Execute query
		executeQuery(NS, model, queryStringDonors);
		
		// -----------------------------
		// 1.4. Número de proyectos
		// -----------------------------
		System.out.println();
		System.out.println("------------------------");
		System.out.println("1.4. Número de proyectos");
		System.out.println("------------------------");
		System.out.println();

		String queryStringNumProyects = "PREFIX a: <" + NS + "> " +
				"PREFIX rdf: <" + rdf + "> " +
				"PREFIX xsd: <" + xsd + "> " +
				"SELECT (COUNT( * ) as ?numberOfProyects) \n" +
				"WHERE" +
				"{" +
					"?id rdf:type a:Project ." +
				"}";
		
		// Execute query
		executeQuery(NS, model, queryStringNumProyects);
		
		// -----------------------------
		// 1.5. Número de laboratorios
		// -----------------------------
		System.out.println();
		System.out.println("------------------------");
		System.out.println("1.5. Número de laboratorios");
		System.out.println("------------------------");
		System.out.println();
		
		String queryStringNumLabs = "PREFIX a: <" + NS + "> " +
				"PREFIX rdf: <" + rdf + "> " +
				"PREFIX xsd: <" + xsd + "> " +
				"SELECT (COUNT( DISTINCT ?lab) AS ?numOfLabs) \n" +
				"WHERE" +
				"{" +
					"?id rdf:type a:Project ;" +
					"    a:SPR.hasLaboratory ?lab ." +
				"}";
		
		// Execute query
		executeQuery(NS, model, queryStringNumLabs);
		
		// -----------------------------
		// 1.6. Número de especímenes
		// -----------------------------
		System.out.println();
		System.out.println("------------------------");
		System.out.println("1.6. Número de especímenes");
		System.out.println("------------------------");
		System.out.println();
		
		System.out.println();
		
		String queryStringNumSpecimens = "PREFIX a: <" + NS + "> " +
				"PREFIX rdf: <" + rdf + "> " +
				"PREFIX xsd: <" + xsd + "> " +
				"SELECT (SUM(?specimenCount) AS ?totalSpecimenCount) \n" +
				"WHERE " +
				"{" +
					"?id rdf:type a:Project ;" +
					"    a:PR.hasSpecimenCount ?specimenCount ." +
				"} ";
		
		// Execute query
		executeQuery(NS, model, queryStringNumSpecimens);
		
		System.out.println();

		queryStringNumSpecimens = "PREFIX a: <" + NS + "> " +
				"PREFIX rdf: <" + rdf + "> " +
				"PREFIX xsd: <" + xsd + "> " +
				"SELECT ?title ?specimenCountProjects \n" +
				"WHERE " +
				"{" +
					"?id rdf:type a:Project ;" +
					"    a:SPR.hasProjectTitle ?title ;" +
					"    a:PR.hasSpecimenCount ?specimenCountProjects ." +
				"} " +
				"ORDER BY ?title";	
		
		// Execute query
		executeQuery(NS, model, queryStringNumSpecimens);
		
		// -----------------------------
		// 1.7. Tamaño total de ficheros
		// -----------------------------
		System.out.println();
		System.out.println("------------------------");
		System.out.println("1.5. Tamaño total de ficheros");
		System.out.println("------------------------");
		System.out.println();
		
		String queryStringTotalSize = "PREFIX a: <" + NS + "> " +
				"PREFIX rdf: <" + rdf + "> " +
				"PREFIX xsd: <" + xsd + "> " +
				"SELECT (SUM(?size)/(1024 * 1024) as ?totalSizeProjects) \n" +
				"WHERE" +
				"{" +
					"?id rdf:type a:Project ;" +
					"    a:SPR.hasTotalSizeOfFilesInMB ?size ." +
					"FILTER (?size != -1) " +
				"}";
		
		// Execute query
		executeQuery(NS, model, queryStringTotalSize);
		
		// -----------------------------
		// 1.7. Número de células por objeto de estudio
		// -----------------------------
		System.out.println();
		System.out.println("------------------------");
		System.out.println("1.7. Número de células por objeto de estudio");
		System.out.println("------------------------");
		System.out.println();
		
		String queryStringNumCellsPerObject = "PREFIX a: <" + NS + "> " +
				"PREFIX rdf: <" + rdf + "> " +
				"PREFIX xsd: <" + xsd + "> " +
				"SELECT ?object (IF(SUM(?numCells) = 0, \"unspecified\", SUM(?numCells) / 1000) AS ?numTotalCells) \n" +
				"WHERE" +
				"{" +
					"?id rdf:type a:Specimen ;" +
					"    a:SPR.hasOrganismPart ?object ;" +
					"    a:SPR.hasTotalCellCount ?numCells ." +
					"FILTER (?numCells != -1)" +
				"}" +
				"GROUP BY ?object " +
				"ORDER BY ?object";
		
		// Execute query
		executeQuery(NS, model, queryStringNumCellsPerObject);
		
		// -----------------------------
		// 1.8. Número de donantes por proyecto
		// -----------------------------
		System.out.println();
		System.out.println("------------------------");
		System.out.println("1.8. Número de donantes por proyecto");
		System.out.println("------------------------");
		System.out.println();
		
		String queryStringDonorsPerProject = "PREFIX a: <" + NS + "> " +
				"PREFIX rdf: <" + rdf + "> " +
				"PREFIX xsd: <" + xsd + "> " +
				"SELECT ?project ( IF( ?donorCount = 0, \"unspecified\", ?donorCount ) AS ?totalDonorCount ) \n" +
				"WHERE" +
				"{" +
					"?id rdf:type a:Project ;" +
					"    a:SPR.hasProjectTitle ?project ;" +
					"    a:PR.hasDonorCount ?donorCount ." +
					"FILTER (?donorCount != -1) " +
				"}" +
				"ORDER BY ?project";
		
		// Execute query
		executeQuery(NS, model, queryStringDonorsPerProject);
		
		// -----------------------------
		// 1.9. Número de células por proyecto
		// -----------------------------
		System.out.println();
		System.out.println("------------------------");
		System.out.println("1.9. Número de células por proyecto");
		System.out.println("------------------------");
		System.out.println();
		
		String queryStringNumCellsPerProject = "PREFIX a: <" + NS + "> " +
				"PREFIX rdf: <" + rdf + "> " +
				"PREFIX xsd: <" + xsd + "> " +
				"SELECT ?project ( IF( ?numCells = 0, \"unspecified\", ?numCells) AS ?numTotalCells) \n" +
				"WHERE" +
				"{" +
					"?id rdf:type a:Project ;" +
					"    a:SPR.hasProjectTitle ?project ;" +
					"    a:SPR.hasTotalCellCount ?numCells ." +
					"FILTER (?numCells != -1) " +
				"}" +
				"ORDER BY ?project";
		
		// Execute query
		executeQuery(NS, model, queryStringNumCellsPerProject);
		
		/* 2. Consultas sencillas que podríamos hacer en HCA
		 * 
		 * a) Para conocer el repositorio
		 * 
		 * ¿Cuál es el laboratorio que más proyectos tiene y cuál es su principal órgano estudiado?
		 * ¿Cuál es el instrumento, el tipo de librería y el protocolo de análisis más utilizados en este repositorio?
		 * ¿De cuántos especímenes tenemos disponibles modelos de órganos y de qué órganos se trata? donde órgano me refiero como término de la HCA.
		 * ¿De cuántos especímenes tenemos disponibles datos de líneas celulares y para qué tipo de células?
		 * ¿Cuál es el órgano para el cual tenemos un mayor número de enfermedades estudiadas?
		 * ¿Cuál es el proyecto que tiene un mayor número de especímenes asociado? ¿y el mayor número de cell counts?
		 * 
		 * b) Como usuario que busca sobre un tema concreto
		 * 
		 * Si me interesa como "órgano" de estudio blood, ¿de qué tipos celulares hay datos de single-cell disponibles en HCA? Si en su lugar  tuviera interés en el "órgano"
		 * immune system, ¿de qué tipos celulares hay datos de single-cell disponibles en HCA? ¿Qué tipos celulares son coincidentes entre ambos órganos?
		 * ¿Sobre qué enfermedades puedo encontrar datos de single-cell para embrión de Homo Sapiens? ¿y Mus Musculus?
		 * ¿Tenemos datos de single-cell disponibles para un tipo de célula que sea específico de decidua y placenta? ¿Cuál es el título y laboratorio del proyecto?
		 */
		
		
		// -----------------------------
		// 2.1. Para conocer el repositorio
		// -----------------------------
		
		// -----------------------------
		// 2.1.1. ¿Cuál es el instrumento, el tipo de librería y el protocolo de análisis más utilizados en este repositorio?
		// -----------------------------
		System.out.println();
		System.out.println("------------------------");
		System.out.println("2.1.1. ¿Cuál es el instrumento, el tipo de librería y el protocolo de análisis más utilizados en este repositorio?");
		System.out.println("------------------------");
		System.out.println();
		
		String queryStringMaxInstrument = "PREFIX a: <" + NS + "> " +
				"PREFIX rdf: <" + rdf + "> " +
				"PREFIX xsd: <" + xsd + "> " +
				"SELECT ?instrument (COUNT(*) as ?numberOfOccurrences) \n" +
				"WHERE" +
				"{" +
					"?id rdf:type a:Project ;" +
					"    a:SPR.hasInstrument ?instrument ." +
				"}" +
				"GROUP BY ?instrument \n" +
				"ORDER BY DESC(?numberOfOccurrences) \n" +
				"LIMIT 1";
		
		// Execute query
		executeQuery(NS, model, queryStringMaxInstrument);
		
		String queryStringMaxLibrary = "PREFIX a: <" + NS + "> " +
				"PREFIX rdf: <" + rdf + "> " +
				"PREFIX xsd: <" + xsd + "> " +
				"SELECT ?library (COUNT(*) as ?numberOfOccurrences) \n" +
				"WHERE" +
				"{" +
					"?id rdf:type a:Project ;" +
					"    a:SPR.hasLibrary ?library ." +
				"}" +
				"GROUP BY ?library \n" +
				"ORDER BY DESC(?numberOfOccurrences) \n" +
				"LIMIT 1";
		
		// Execute query
		executeQuery(NS, model, queryStringMaxLibrary);
		
		String queryStringMaxProtocol = "PREFIX a: <" + NS + "> " +
				"PREFIX rdf: <" + rdf + "> " +
				"PREFIX xsd: <" + xsd + "> " +
				"SELECT ?protocol (COUNT(*) as ?numberOfOccurrences) \n" +
				"WHERE" +
				"{" +
					"?id rdf:type a:Project ;" +
					"    a:SPR.hasAnalysisProtocol ?protocol ." +
				"}" +
				"GROUP BY ?protocol \n" +
				"ORDER BY DESC(?numberOfOccurrences) \n" +
				"LIMIT 1";
		
		// Execute query
		executeQuery(NS, model, queryStringMaxProtocol);
		
		// -----------------------------
		// 2.1.2. ¿De cuántos proyectos tenemos disponibles modelos de órganos y de qué órganos se trata?
		// -----------------------------
		System.out.println();
		System.out.println("------------------------");
		System.out.println("2.1.2. ¿De cuántos proyectos tenemos disponibles modelos de órganos y de qué órganos se trata?");
		System.out.println("------------------------");
		System.out.println();
		
		String queryStringNumberOfModels = "PREFIX a: <" + NS + "> " +
				"PREFIX rdf: <" + rdf + "> " +
				"PREFIX xsd: <" + xsd + "> " +
				"SELECT (COUNT( DISTINCT ?id ) as ?numberOfProjectsWithModels) \n" +
				"WHERE" +
				"{" +
					"?id rdf:type a:Project ;" +
					"    a:SPR.hasModel ?model ." +
				"}";
		
		// Execute query
		executeQuery(NS, model, queryStringNumberOfModels);
		
		System.out.println();
		
		String queryStringModels = "PREFIX a: <" + NS + "> " +
				"PREFIX rdf: <" + rdf + "> " +
				"PREFIX xsd: <" + xsd + "> " +
				"SELECT ?model ( COUNT(*) AS ?numberOfOccurrences) \n" +
				"WHERE" +
				"{" +
					"?id rdf:type a:Project ;" +
					"    a:SPR.hasModel ?model ." +
				"}" +
				"GROUP BY ?model";
		
		// Execute query
		executeQuery(NS, model, queryStringModels);
		
		// -----------------------------
		// 2.1.3. ¿De cuántos proyectos tenemos disponibles datos de líneas celulares?
		// -----------------------------
		System.out.println();
		System.out.println("------------------------");
		System.out.println("2.1.3. ¿De cuántos proyectos tenemos disponibles datos de líneas celulares?");
		System.out.println("------------------------");
		System.out.println();
		
		String queryStringNumberOfCellLines = "PREFIX a: <" + NS + "> " +
				"PREFIX rdf: <" + rdf + "> " +
				"PREFIX xsd: <" + xsd + "> " +
				"SELECT (COUNT( ?cellLine ) as ?numberOfProjectsWithCellLine) \n" +
				"WHERE" +
				"{" +
					"?id rdf:type a:Project ;" +
					"    a:SPR.hasCellLineType ?cellLine ." +
				"}";
		
		// Execute query
		executeQuery(NS, model, queryStringNumberOfCellLines);
		
		// -----------------------------
		// 2.1.4. ¿Cuál es el proyecto que tiene un mayor número de donantes asociado? ¿y el mayor número de cell counts?
		// -----------------------------
		System.out.println();
		System.out.println("------------------------");
		System.out.println("2.1.4. ¿Cuál es el proyecto que tiene un mayor número de donantes asociado? ¿y el mayor número de cell counts?");
		System.out.println("------------------------");
		System.out.println();
		
		String queryStringProjectWithMostSpecimens = "PREFIX a: <" + NS + "> " +
				"PREFIX rdf: <" + rdf + "> " +
				"PREFIX xsd: <" + xsd + "> " +
				"SELECT ?project ?donors \n" +
				"WHERE" +
				"{" +
					"?id rdf:type a:Project ;" +
					"    a:SPR.hasProjectTitle ?project ;" +
					"    a:PR.hasDonorCount ?donors ." +
				"}" +
				"ORDER BY DESC(?donors) \n" +
				"LIMIT 1";
		
		// Execute query
		executeQuery(NS, model, queryStringProjectWithMostSpecimens);
		
		String queryStringProjectWithMostCells = "PREFIX a: <" + NS + "> " +
				"PREFIX rdf: <" + rdf + "> " +
				"PREFIX xsd: <" + xsd + "> " +
				"SELECT ?project ?cells \n" +
				"WHERE" +
				"{" +
					"?id rdf:type a:Project ;" +
					"    a:SPR.hasProjectTitle ?project ;" +
					"    a:SPR.hasTotalCellCount ?cells ." +
				"}" +
				"ORDER BY DESC(?cells) \n" +
				"LIMIT 1";
		
		// Execute query
		executeQuery(NS, model, queryStringProjectWithMostCells);
		
		// -----------------------------
		// 2.2. Como usuario que busca sobre un tema concreto
		// -----------------------------
				
		// -----------------------------
		// 2.2.1 Datos de single cell para hombres de más de 50 años que cuyo organo estudiado es el colon 
		// -----------------------------
		
		System.out.println();
		System.out.println("------------------------");
		System.out.println("2.2.1. Datos de single cell para hombres de más de 50 años que cuyo organo estudiado es el colon");
		System.out.println("------------------------");
		System.out.println();
		
		String queryStringColonMales = "PREFIX a: <" + NS + "> " +
				"PREFIX rdf: <" + rdf + "> " +
				"PREFIX xsd: <" + xsd + "> " +
				"SELECT ?specimen_id \n" +
				"WHERE" +
				"{" +
					"?id rdf:type a:Specimen ;" +
					"    a:SR.hasSpecimenID ?specimen_id ;" +
					"    a:SPR.hasSpecie a:HomoSapiens ;" +
					"    a:SPR.hasOrganismPart a:Colon ;" +
					"    a:SPR.hasBiologicalSex \"male\" ;" +
					"    a:SPR.hasAgeUnit ?unit ;" +
					"    a:SPR.hasMinAge ?min_age ." +
					"FILTER (?unit = \"y\" || ?unit = \"year\")" +
					"FILTER ( ?min_age > 50 )" +
				"}";
		
		// Execute query
		executeQuery(NS, model, queryStringColonMales);
			
		// -----------------------------
		// 2.2.2 ¿Que partes del organismo estudian los proyectos de líneas celulares y que proyectos son? 
		// -----------------------------
		
		System.out.println();
		System.out.println("------------------------");
		System.out.println("2.2.2. ¿Que partes del organismo estudian los proyectos de líneas celulares? ");
		System.out.println("------------------------");
		System.out.println();
		
		String queryCellLinesOrgans = "PREFIX a: <" + NS + "> " +
				"PREFIX rdf: <" + rdf + "> " +
				"PREFIX xsd: <" + xsd + "> " +
				"SELECT ?project ?organism_part \n" +
				"WHERE" +
				"{" +
					"?id rdf:type a:Project ;" +
					"    a:SPR.hasProjectTitle ?project ;" +
					"    a:SPR.hasOrganismPart ?organism_part ;" +
					"    a:SPR.hasSampleType a:CellLines ." +
				"}";
		
		// Execute query
		executeQuery(NS, model, queryCellLinesOrgans);
		
		// -----------------------------
		// 2.2.3 ¿Qué tipos celulares, y de que proyectos, se han seleccionado para estudiar la diabete de tipo 2? 
		// -----------------------------
		
		System.out.println();
		System.out.println("------------------------");
		System.out.println("2.2.3. ¿Qué tipos celulares, y de que proyectos, se han seleccionado para estudiar la diabete de tipo 2?");
		System.out.println("------------------------");
		System.out.println();
		
		String queryStringDiabetesCellType = "PREFIX a: <" + NS + "> " +
				"PREFIX rdf: <" + rdf + "> " +
				"PREFIX xsd: <" + xsd + "> " +
				"SELECT DISTINCT ?project ?cell_type \n" +
				"WHERE" +
				"{" +
					"?id rdf:type a:Specimen ;" +
					"    a:SPR.hasDiseaseStatus a:Type2DiabetesMellitus ;" +
					"    a:SPR.hasProjectTitle ?project ;" +
					"    a:SPR.hasSelectedCellType ?cell_type ." +
				"}";
		
		// Execute query
		executeQuery(NS, model, queryStringDiabetesCellType);
		
		/* 3. Consultas que no podríamos hacer en HCA pero sí en nuestra ontología
		 * 
		 * a) Para demostrar que no arrastramos inconsistencias
		 *	
		 *	En el ejemplo que os mostré en la reunión (está en el pdf que os dejé), veíamos que las partes de órganos descritas para riñón incluían cortex y cortex of kidney, 
		 *  sin embargo, cortex es específico de cerebro. Por lo tanto, la primera consulta que podríamos hacer es que conocer las partes de riñón y cerebro según nuestra 
		 *  ontología.
         *
		 * b) Como usuarios que buscan sobre un tema concreto
		 *  
		 *	¿De cuántos especímenes tendríamos disponibles datos de single-cell cuyo tipo celular pertenezca a la clase leukocytes?
		 *	¿De qué especies tengo datos de single-cell que pertenezcan al reino Plants?
		 * 	¿De cuántos especímenes tenemos disponibles datos de single-cell cuyos donantes (especímenes, indistintamente) tengan la edad expresada en días? ¿De qué SampleType 
		 *  se trata? SI es línea celular, ¿de qué tipo se trata?
		 *	¿De cuántos especímenes disponemos los metadatos, la matriz y los resultados? ¿Cuál son los objectos de estudio?
		 *	¿De qué enfermedades tenemos disponible datos de single-cell donde el "órgano" afectado sea el corazón o, directamente, el sistema circulatorio? ¿Cuáles son estas 
		 *  enfermedades?
		 *	¿Qué proyectos tenemos donde los especímenes estén afectados por una enfermedad clasificada como metabólica y hereditaria? ¿Qué órgano se encuentra afectado? ¿De 
		 *  qué sistema forma parte?
		 */
		
		// -----------------------------
		// 3.1. Demostrar la importancia de hacer búsquedas a nivel de especímenes en lugar de proyectos
		// -----------------------------
		
		// -----------------------------
		// 3.1.1. Si me interesa como "órgano" de estudio blood, ¿de qué tipos celulares hay datos de single-cell disponibles en HCA?
		// -----------------------------
		System.out.println();
		System.out.println("------------------------");
		System.out.println("3.1.1. Si me interesa como \"órgano\" de estudio blood, ¿de qué tipos celulares hay datos de single-cell disponibles en HCA?");
		System.out.println("------------------------");
		System.out.println();

		System.out.println("Busqueda a nivel de proyecto:");
		
		String queryStringBloodCellType = "PREFIX a: <" + NS + "> " +
				"PREFIX rdf: <" + rdf + "> " +
				"PREFIX xsd: <" + xsd + "> " +
				"SELECT DISTINCT ?cellType \n" +
				"WHERE" +
				"{" +
					"?id rdf:type a:Project ;" +
					"    a:SPR.hasOrganismPart a:Blood ;" +
					"    a:SPR.hasSelectedCellType ?cellType ." +
				"}";
		
		// Execute query
		executeQuery(NS, model, queryStringBloodCellType);
		
		System.out.println();
		System.out.println("Busqueda a nivel de espécimen:");
		
		queryStringBloodCellType = "PREFIX a: <" + NS + "> " +
				"PREFIX rdf: <" + rdf + "> " +
				"PREFIX xsd: <" + xsd + "> " +
				"SELECT DISTINCT ?cellType \n" +
				"WHERE" +
				"{" +
					"?id rdf:type a:Specimen ;" +
					"    a:SPR.hasOrganismPart a:Blood ;" +
					"    a:SPR.hasSelectedCellType ?cellType ." +
				"}";
		
		// Execute query
		executeQuery(NS, model, queryStringBloodCellType);
		
		// -----------------------------
		// 3.1.2. Interés en el "órgano" immune system, ¿de qué tipos celulares hay datos de single-cell disponibles en HCA?
		// -----------------------------
		System.out.println();
		System.out.println("------------------------");
		System.out.println("3.1.2. Interés en el \"órgano\" immune system, ¿de qué tipos celulares hay datos de single-cell disponibles en HCA?");
		System.out.println("------------------------");
		System.out.println();
		
		System.out.println("Busqueda a nivel de proyecto:");
		
		String queryStringImmuneSystemCellType = "PREFIX a: <" + NS + "> " +
				"PREFIX rdf: <" + rdf + "> " +
				"PREFIX xsd: <" + xsd + "> " +
				"SELECT DISTINCT ?cellType \n" +
				"WHERE" +
				"{" +
					"?id rdf:type a:Project ;" +
					"    a:SPR.hasOrganismPart a:ImmuneSystem ;" +
					"    a:SPR.hasSelectedCellType ?cellType ." +
				"}";
		
		// Execute query
		executeQuery(NS, model, queryStringImmuneSystemCellType);

		System.out.println();
		System.out.println("Busqueda a nivel de espécimen:");
		
		queryStringImmuneSystemCellType = "PREFIX a: <" + NS + "> " +
				"PREFIX rdf: <" + rdf + "> " +
				"PREFIX xsd: <" + xsd + "> " +
				"SELECT DISTINCT ?cellType \n" +
				"WHERE" +
				"{" +
					"?id rdf:type a:Specimen ;" +
					"    a:SPR.hasOrganismPart a:ImmuneSystem ;" +
					"    a:SPR.hasSelectedCellType ?cellType ." +
				"}";
		
		// Execute query
		executeQuery(NS, model, queryStringImmuneSystemCellType);
		
		// -----------------------------
		// 3.1.3. ¿Qué tipos celulares son coincidentes entre ambos órganos?
		// -----------------------------
		System.out.println();
		System.out.println("------------------------");
		System.out.println("3.1.3. ¿Qué tipos celulares son coincidentes entre ambos órganos?");
		System.out.println("------------------------");
		System.out.println();

		System.out.println("Busqueda a nivel de proyecto:");

		String queryStringCommonCellType = "PREFIX a: <" + NS + "> " +
				"PREFIX rdf: <" + rdf + "> " +
				"PREFIX xsd: <" + xsd + "> " +
				"SELECT DISTINCT ?cellType " +
				"WHERE " +
				"{ " +
					"{ " +
						"?id1 rdf:type a:Project ;" +
						"     a:SPR.hasOrganismPart a:Blood ;" +
						"     a:SPR.hasSelectedCellType ?cellType ." +
					"} " +
					"{" +
						"?id2 rdf:type a:Project ;" +
						"     a:SPR.hasOrganismPart a:ImmuneSystem ;" +
						"     a:SPR.hasSelectedCellType ?cellType ." +
					"}" +
				"}";
		
		// Execute query
		executeQuery(NS, model, queryStringCommonCellType);
		
		System.out.println();
		System.out.println("Busqueda a nivel de espécimen:");
		
		queryStringCommonCellType = "PREFIX a: <" + NS + "> " +
				"PREFIX rdf: <" + rdf + "> " +
				"PREFIX xsd: <" + xsd + "> " +
				"SELECT DISTINCT ?cellType " +
				"WHERE " +
				"{ " +
					"{ " +
						"?id1 rdf:type a:Specimen ;" +
						"     a:SPR.hasOrganismPart a:Blood ;" +
						"     a:SPR.hasSelectedCellType ?cellType ." +
					"} " +
					"{" +
						"?id2 rdf:type a:Specimen ;" +
						"     a:SPR.hasOrganismPart a:ImmuneSystem ;" +
						"     a:SPR.hasSelectedCellType ?cellType ." +
					"}" +
				"}";
		
		// Execute query
		executeQuery(NS, model, queryStringCommonCellType);

		// -----------------------------
		// 3.2. Demostrar que no arrastramos inconsistencias
		// -----------------------------
		
		// -----------------------------
		// 3.2.1. Partes del riñón
		// -----------------------------
		System.out.println();
		System.out.println("------------------------");
		System.out.println("3.2.1. Partes del riñón.");
		System.out.println("------------------------");
		System.out.println();
		
		String queryStringKidneyParts = "PREFIX a: <" + NS + "> " +
				"PREFIX rdf: <" + rdf + "> " +
				"PREFIX xsd: <" + xsd + "> " +
				"SELECT DISTINCT ?kidneyParts " +
				"WHERE " +
				"{ " +
					"?kidneyParts a:OR.isOrganPartOf a:Kidney ;" +
				"}";
		
		// Execute query
		executeQuery(NS, model, queryStringKidneyParts);
		
		// -----------------------------
		// 3.2.2. Partes del cerebro
		// -----------------------------
		System.out.println();
		System.out.println("------------------------");
		System.out.println("3.2.2. Partes del cerebro.");
		System.out.println("------------------------");
		System.out.println();
		
		String queryStringBrainParts = "PREFIX a: <" + NS + "> " +
				"PREFIX rdf: <" + rdf + "> " +
				"PREFIX xsd: <" + xsd + "> " +
				"SELECT DISTINCT ?brainParts " +
				"WHERE " +
				"{ " +
					"?brainParts a:OR.isOrganPartOf a:Brain ;" +
				"}";
		
		// Execute query
		executeQuery(NS, model, queryStringBrainParts);
		
		// -----------------------------
		// 3.3. Como usuarios que buscan sobre un tema concreto
		// -----------------------------
		
		// -----------------------------
		// 3.3.1. ¿De qué tipos celulares tendríamos disponibles datos de single-cell cuyo tipo celular pertenezca a la clase leukocytes?
		// -----------------------------
		System.out.println();
		System.out.println("------------------------");
		System.out.println("3.3.1. ¿De qué tipos celulares tendríamos disponibles datos de single-cell cuyo tipo celular pertenezca a la clase leukocytes?.");
		System.out.println("------------------------");
		System.out.println();
		
		String queryStringLeukocytesSpecimens = "PREFIX a: <" + NS + "> " +
				"PREFIX rdf: <" + rdf + "> " +
				"PREFIX xsd: <" + xsd + "> " +
				"SELECT DISTINCT ?cellType " +
				"WHERE " +
				"{ " +
					"?id rdf:type a:Specimen ;" +
					"    a:SPR.hasSelectedCellType ?cellType ." +
					"?cellType rdf:type a:Leukocyte ." +
				"}";
		
		// Execute query
		executeQuery(NS, model, queryStringLeukocytesSpecimens);
		
		// -----------------------------
		// 3.3.2. ¿De qué especies tengo datos de single-cell que pertenezcan al reino Plants?
		// -----------------------------
		System.out.println();
		System.out.println("------------------------");
		System.out.println("3.3.2. ¿De qué especies tengo datos de single-cell que pertenezcan al reino Plants?");
		System.out.println("------------------------");
		System.out.println();
		
		String queryStringPlantsSpecies = "PREFIX a: <" + NS + "> " +
				"PREFIX rdf: <" + rdf + "> " +
				"PREFIX xsd: <" + xsd + "> " +
				"SELECT ?plantSpecie " +
				"WHERE " +
				"{ " +
					"?plantSpecie a:OR.belongsToKingdom a:Plants ." +
				"}";
		
		// Execute query
		executeQuery(NS, model, queryStringPlantsSpecies);
		
		// -----------------------------
		// 3.3.3. ¿De cuántos especímenes tenemos disponibles datos de single-cell cuyos donantes (especímenes, indistintamente) tengan la edad expresada en días?
		// -----------------------------
		System.out.println();
		System.out.println("------------------------");
		System.out.println("3.3.3. ¿De cuántos especímenes tenemos disponibles datos de single-cell cuyos donantes (especímenes, indistintamente) tengan la edad expresada en días?");
		System.out.println("------------------------");
		System.out.println();
		
		String queryStringSpecimensWithAgeInDays = "PREFIX a: <" + NS + "> " +
				"PREFIX rdf: <" + rdf + "> " +
				"PREFIX xsd: <" + xsd + "> " +
				"SELECT (COUNT(*) AS ?specimensWithAgeInDays) " +
				"WHERE " +
				"{ " +
					"?id rdf:type a:Specimen ;" +
					"    a:SPR.hasAgeUnit ?unit ." +
					"FILTER (?unit = \"d\" || ?unit = \"day\")" +
				"}";
		
		// Execute query
		executeQuery(NS, model, queryStringSpecimensWithAgeInDays);
		
		
				
		// -----------------------------
		// 3.3.4. ¿De qué SampleType se trata?
		// -----------------------------
		System.out.println();
		System.out.println("------------------------");
		System.out.println("3.3.4. ¿De qué SampleType se trata?");
		System.out.println("------------------------");
		System.out.println();
		
		String queryStringSpecimensWithAgeInDaysSampleType = "PREFIX a: <" + NS + "> " +
				"PREFIX rdf: <" + rdf + "> " +
				"PREFIX xsd: <" + xsd + "> " +
				"SELECT DISTINCT ?sampleType \n" +
				"WHERE \n" +
				"{ " +
					"?id rdf:type a:Specimen ;" +
					"    a:SPR.hasAgeUnit ?unit ;" +
					"    a:SPR.hasSampleType ?sampleType ." +
					"FILTER (?unit = \"d\" || ?unit = \"day\") " +
				"}";
		
		// Execute query
		executeQuery(NS, model, queryStringSpecimensWithAgeInDaysSampleType);
		
		// -----------------------------
		// 3.3.5. ¿De cuántos proyectos disponemos los metadatos, la matriz y los resultados?
		// -----------------------------
		System.out.println();
		System.out.println("------------------------");
		System.out.println("3.3.5. ¿De cuántos especímenes disponemos los metadatos, la matriz y los resultados?");
		System.out.println("------------------------");
		System.out.println();
		
		String queryStringSpecimensWith3Types= "PREFIX a: <" + NS + "> " +
				"PREFIX rdf: <" + rdf + "> " +
				"PREFIX xsd: <" + xsd + "> " +
				"SELECT (COUNT(*) AS ?numProyects) \n" +
				"WHERE \n" +
				"{ " +
					"?id rdf:type a:Project ;" +
					"    a:PR.hasAvailableDownloadsType \"matrix\" ;" +
					"    a:PR.hasAvailableDownloadsType \"results\" ;" +
					"    a:PR.hasAvailableDownloadsType \"metadata\" ." +
				"}";		
		// Execute query
		executeQuery(NS, model, queryStringSpecimensWith3Types);
		
		// -----------------------------
		// 3.3.6. ¿Cuál son las partes del organismo?
		// -----------------------------
		System.out.println();
		System.out.println("------------------------");
		System.out.println("3.3.6. ¿Cuál son las partes del organismo?");
		System.out.println("------------------------");
		System.out.println();
		
		String queryStringObjectsWith3Types= "PREFIX a: <" + NS + "> " +
				"PREFIX rdf: <" + rdf + "> " +
				"PREFIX xsd: <" + xsd + "> " +
				"SELECT ?object (COUNT(*) AS ?numProjects) \n" +
				"WHERE \n" +
				"{ " +
					"?id rdf:type a:Project ;" +
					"    a:PR.hasAvailableDownloadsType \"matrix\" ;" +
					"    a:PR.hasAvailableDownloadsType \"results\" ;" +
					"    a:PR.hasAvailableDownloadsType \"metadata\" ;" +
					"    a:SPR.hasOrganismPart ?object ." +
				"} " +
				"GROUP BY ?object";		
		// Execute query
		executeQuery(NS, model, queryStringObjectsWith3Types);
		
		// -----------------------------
		// 3.3.7. ¿De qué enfermedades tenemos disponible datos de single-cell donde el "órgano" afectado sea el corazón o, directamente, el sistema circulatorio? ¿Cuáles son estas 
		//        enfermedades?
		// -----------------------------
		System.out.println();
		System.out.println("------------------------");
		System.out.println("3.3.7. ¿De qué enfermedades tenemos disponible datos de single-cell donde el \"órgano\" afectado sea el corazón o, directamente, el sistema circulatorio? ¿Cuáles son estas enfermedades?");
		System.out.println("------------------------");
		System.out.println();
		
		String queryStringDiseaseHeart= "PREFIX a: <" + NS + "> " +
				"PREFIX rdf: <" + rdf + "> " +
				"PREFIX xsd: <" + xsd + "> " +
				"SELECT DISTINCT ?disease \n" +
				"WHERE \n" +
				"{ " +
					"{ " +
						"?disease rdf:type a:DiseaseStatus ;" +
						"         a:OR.hasAffected a:Heart ." +
					"} " +
					"UNION " +
					"{ " +
						"?disease rdf:type a:Specimen ;" +
						"         a:OR.hasAffected a:CardiovascularSystem ." +
					"} " +
				"}";		
		// Execute query
		executeQuery(NS, model, queryStringDiseaseHeart);
		
		// -----------------------------
		// 3.3.8. ¿Qué proyectos tenemos donde los especímenes estén afectados por una enfermedad clasificada como metabólica y hereditaria?
		// -----------------------------
		System.out.println();
		System.out.println("------------------------");
		System.out.println("3.3.8. ¿Qué proyectos tenemos donde los especímenes estén afectados por una enfermedad clasificada como metabólica y hereditaria?");
		System.out.println("------------------------");
		System.out.println();
		
		String queryStringProyectsWithMetabolicOrHereditary= "PREFIX a: <" + NS + "> " +
				"PREFIX rdf: <" + rdf + "> " +
				"PREFIX xsd: <" + xsd + "> " +
				"SELECT DISTINCT ?proyect ?diseaseStatus \n" +
				"WHERE \n" +
				"{ " +
					"{ " +
						"?id rdf:type a:Specimen ;" +
						"    a:SPR.hasProjectTitle ?proyect ;" +
						"    a:SPR.hasDiseaseStatus ?diseaseStatus ." +
						"?diseaseStatus rdf:type a:DiaseaseOfMetabolism ." +
					"} " +
					"{ " +
						"?id rdf:type a:Specimen ;" +
						"    a:SPR.hasProjectTitle ?proyect ;" +
						"    a:SPR.hasDiseaseStatus ?diseaseStatus ." +
						"?diseaseStatus rdf:type a:GeneticDisease ." +
					"} " +
				"}";		
		
		// Execute query
		executeQuery(NS, model, queryStringProyectsWithMetabolicOrHereditary);
		
		// -----------------------------
		// 3.3.9. ¿Qué órgano se encuentra afectado según nuestra onlogía?
		// -----------------------------
		System.out.println();
		System.out.println("------------------------");
		System.out.println("3.3.9. ¿Qué órgano se encuentra afectado según nuestra ontología?");
		System.out.println("------------------------");
		System.out.println();
		
		String queryStringOrgansWithMetabolicOrHereditary = "PREFIX a: <" + NS + "> " +
				"PREFIX rdf: <" + rdf + "> " +
				"PREFIX xsd: <" + xsd + "> " +
				"SELECT DISTINCT ?diseaseStatus ?organ \n" +
				"WHERE \n" +
				"{ " +
					"{ " +
						"?id rdf:type a:Specimen ;" +
						"    a:SPR.hasDiseaseStatus ?diseaseStatus ." +
						"?diseaseStatus rdf:type a:DiaseaseOfMetabolism ;" +
						"               a:OR.hasAffected ?organ ." +
					"} " +
					"{ " +
						"?id rdf:type a:Specimen ;" +
						"    a:SPR.hasDiseaseStatus ?diseaseStatus ." +
						"?diseaseStatus rdf:type a:GeneticDisease ;" +
						"               a:OR.hasAffected ?organ ." +
					"} " +
				"}";
		
		// Execute query
		executeQuery(NS, model, queryStringOrgansWithMetabolicOrHereditary);
		
		// -----------------------------
		// 3.3.10. ¿De qué sistema forma parte?
		// -----------------------------
		System.out.println();
		System.out.println("------------------------");
		System.out.println("3.3.10. ¿De qué sistema forma parte?");
		System.out.println("------------------------");
		System.out.println();
		
		String queryStringSystemsWithMetabolicOrHereditary = "PREFIX a: <" + NS + "> " +
				"PREFIX rdf: <" + rdf + "> " +
				"PREFIX xsd: <" + xsd + "> " +
				"SELECT DISTINCT ?diseaseStatus ?object ?system \n" +
				"WHERE \n" +
				"{ " +
					"{ " +
						"?id rdf:type a:Specimen ;" +
						"    a:SPR.hasDiseaseStatus ?diseaseStatus ." +
						"?diseaseStatus rdf:type a:DiaseaseOfMetabolism ;" +
						"               a:OR.hasAffected ?organ ." +
						"?organ a:OR.isPartOfSystem ?system ." +
					"} " +
					"{ " +
						"?id rdf:type a:Specimen ;" +
						"    a:SPR.hasDiseaseStatus ?diseaseStatus ." +
						"?diseaseStatus rdf:type a:GeneticDisease ;" +
						"               a:OR.hasAffected ?organ ." +
						"?organ a:OR.isPartOfSystem ?system ." +
					"} " +
				"}";		
		
		// Execute query
		executeQuery(NS, model, queryStringSystemsWithMetabolicOrHereditary);
		
		// -----------------------------
		// 3.3.11. ¿Cuál es la institución que más proyectos tiene y cuál es su principal órgano estudiado?
		// -----------------------------
		System.out.println();
		System.out.println("------------------------");
		System.out.println("3.3.11. ¿Cuál es la institución que más proyectos tiene y cuál es su principal órgano estudiado?");
		System.out.println("------------------------");
		System.out.println();
		
		String queryStringMaxLab = "PREFIX a: <" + NS + "> " +
				"PREFIX rdf: <" + rdf + "> " +
				"PREFIX xsd: <" + xsd + "> " +
				"SELECT ?institution (COUNT(DISTINCT ?project) as ?numberOfProyects) \n" +
				"WHERE" +
				"{" +
					"?id rdf:type a:Project ;" +
					"    a:PR.hasInstitution ?institution ;" +
					"    a:SPR.hasProjectTitle ?project ." +
				"}" +
				"GROUP BY ?institution \n" +
				"ORDER BY DESC(?numberOfProyects) \n" +
				"LIMIT 1";
		
		// Execute query
		executeQuery(NS, model, queryStringMaxLab);
		
		System.out.println();
		
		String queryStringMaxLabOrgan = "PREFIX a: <" + NS + "> " +
				"PREFIX rdf: <" + rdf + "> " +
				"PREFIX xsd: <" + xsd + "> " +
				"SELECT ?organ (COUNT(*) as ?numberOfProyects) \n" +
				"WHERE" +
				"{" +
					"?id rdf:type a:Project ;" +
					"    a:PR.hasInstitution \"EMBL-EBI\" ;" +
					"    a:SPR.hasOrganismPart ?organ ." +
				"}" +
				"GROUP BY ?organ \n" +
				"ORDER BY DESC(?numberOfProyects) \n" +
				"LIMIT 1";
		
		// Execute query
		executeQuery(NS, model, queryStringMaxLabOrgan);
		
		// -----------------------------
		// 3.3.12. ¿Cuál es el órgano para el cual tenemos un mayor número de enfermedades estudiadas?
		// -----------------------------
		System.out.println();
		System.out.println("------------------------");
		System.out.println("3.3.12. ¿Cuál es el órgano para el cual tenemos un mayor número de enfermedades estudiadas?");
		System.out.println("------------------------");
		System.out.println();
		
		String queryStringDiseasesPerOrgan = "PREFIX a: <" + NS + "> " +
				"PREFIX rdf: <" + rdf + "> " +
				"PREFIX xsd: <" + xsd + "> " +
				"SELECT ?organismPart (COUNT( DISTINCT ?disease) AS ?numberOfDiseases ) \n" +
				"WHERE" +
				"{" +
					"?id rdf:type a:Specimen ;" +
					"    a:SPR.hasDiseaseStatus ?disease ;" +
					"    a:SPR.hasOrganismPart ?organismPart ." +
					"?organismPart a:OR.isAffectedInDisease ?disease ." +
				"}" +
				"GROUP BY ?organismPart \n" +
				"ORDER BY DESC(?numberOfDiseases) \n" +
				"LIMIT 1";
		
		// Execute query
		executeQuery(NS, model, queryStringDiseasesPerOrgan);
		
		System.out.println();
		
		queryStringDiseasesPerOrgan = "PREFIX a: <" + NS + "> " +
				"PREFIX rdf: <" + rdf + "> " +
				"PREFIX xsd: <" + xsd + "> " +
				"SELECT DISTINCT ?disease \n" +
				"WHERE" +
				"{" +
					"?id rdf:type a:Specimen ;" +
					"    a:SPR.hasDiseaseStatus ?disease ;" +
					"    a:SPR.hasOrganismPart a:Kidney ." +
					"a:Kidney a:OR.isAffectedInDisease ?disease ." +
				"}";
		
		// Execute query
		executeQuery(NS, model, queryStringDiseasesPerOrgan);
		
		// -----------------------------
		// 3.3.13. ¿Tenemos datos de single-cell disponibles para un tipo de célula que sea específico de decidua y placenta?
		// -----------------------------
		System.out.println();
		System.out.println("------------------------");
		System.out.println("3.3.13. ¿Tenemos datos de single-cell disponibles para un tipo de célula que sea específico de decidua y placenta?");
		System.out.println("------------------------");
		System.out.println();
		
		String queryStringCellTypeOnlyDeciduaPlacenta = "PREFIX a: <" + NS + "> " +
				"PREFIX rdf: <" + rdf + "> " +
				"PREFIX xsd: <" + xsd + "> " +
				"SELECT DISTINCT ?cellType " +
				"WHERE " +
				"{ " +
					"{ " +
						"?id1 rdf:type a:Specimen ;" +
						"     a:SPR.hasOrganismPart a:Decidua ;" +
						"     a:SPR.hasSelectedCellType ?cellType ." +
					"} " +
					"{" +
						"?id2 rdf:type a:Specimen ;" +
						"     a:SPR.hasOrganismPart a:Placenta ;" +
						"     a:SPR.hasSelectedCellType ?cellType ." +
					"}" +
					"{" +
						"?id rdf:type a:Specimen ;"	+
						"    a:SPR.hasSelectedCellType ?cellType ;" +
						"    a:SPR.hasOrganismPart ?organism_part ." +
						"FILTER ( ?organism_part = a:Decidua || ?organism_part = a:Placenta )" +
					"}" +
				"}";
		
		// Execute query
		executeQuery(NS, model, queryStringCellTypeOnlyDeciduaPlacenta);
	}

}
