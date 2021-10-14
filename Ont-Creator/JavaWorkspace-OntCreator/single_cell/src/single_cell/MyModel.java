package single_cell;

import java.io.FileInputStream;
import java.io.FileNotFoundException;
import java.io.FileOutputStream;
import java.io.FileWriter;
import java.io.IOException;
import java.util.HashMap;
import java.util.Iterator;

import org.apache.commons.lang3.ArrayUtils;
import org.apache.jena.ontology.Individual;
import org.apache.jena.ontology.ObjectProperty;
import org.apache.jena.ontology.OntClass;
import org.apache.jena.ontology.OntModel;
import org.apache.jena.rdf.model.InfModel;
import org.apache.jena.rdf.model.Model;
import org.apache.jena.rdf.model.ModelFactory;
import org.apache.jena.rdf.model.Property;
import org.apache.jena.rdf.model.Resource;
import org.apache.jena.rdf.model.ResourceFactory;
import org.apache.jena.rdf.model.Statement;
import org.apache.jena.reasoner.ValidityReport;
import org.apache.jena.reasoner.ValidityReport.Report;
import org.apache.jena.vocabulary.ReasonerVocabulary;

import openllet.jena.PelletReasonerFactory;
 

public class MyModel {
	private InfModel model;
	private String NS;
	private FileInputStream inputStream;
	private FileOutputStream outputStream;

	private HashMap<String, Resource> classMap;
	private HashMap<String, Property> dataPropertyMap;
	private HashMap<String, Property> objectPropertyMap;
	private HashMap<String, Property> anotationPropertyMap;
	private HashMap<String, Resource> individualMap;

	public static final String PREDICATE_CLASS = "Type";
	public static final String SPECIMEN_CLASS = "Specimen";
	public static final String PROJECT_CLASS = "Project";

	private final String validateLogFileName = "../../../SingleCell-Files/validateLog.txt";
	
	public static final String[] OBJECT_PROPERTIES = new String[] {
			"SPR.hasAnalysisProtocol",
			"SPR.hasBiopsySite",
			"SPR.hasCellLineType",
			"SPR.hasDisease",
			"SPR.hasInstrument",
			"SPR.hasLibrary",
			"SPR.hasModel",
			"SPR.hasOrganismPart",
			"SPR.hasPreservation",
			"SPR.hasCellType",
			"SPR.hasSpecie",
			"SPR.hasSampleStatus",
	};
	
	public static final String[] SPR_DATA_PROPERTIES = new String[] {
			"SPR.hasSex",
			"SPR.hasPhenotype",
			"SPR.hasTotalCellCount",
			"SPR.isPairedEnd",
			"SPR.hasGrowthCondition",
			"SPR.hasSampleType",
			"SPR.hasNucleicAcidSource",
	};
	
	public static final String[] SR_DATA_PROPERTIES = new String[] {
			"SPR.hasAgeUnit",
			"SPR.hasMaxAge",
			"SPR.hasMinAge",
	};
	
	public static final String[] PR_DATA_PROPERTIES = new String[] {
			"PR.hasDonorCount",
			"PR.hasExperimentalFactor",
			"PR.hasSpecimenCount",
	};
	
	public static final String[] SPECIMEN_DATA_PROPERTIES = ArrayUtils.addAll(SPR_DATA_PROPERTIES, SR_DATA_PROPERTIES);
	public static final String[] PROJECT_DATA_PROPERTIES = ArrayUtils.addAll(SPR_DATA_PROPERTIES, PR_DATA_PROPERTIES);
	
	public static final String[] SPR_ANOTATION_PROPERTIES = new String[] {
			"SPR.hasLaboratory",
			"SPR.hasProjectShortName",
			"SPR.hasProjectTitle",
			"SPR.isPartOfCollection",
			"SPR.isPartOfRepository",
			"SPR.hasClusteringLink",
            "SPR.hasExperimentDesignLink",
            "SPR.hasExperimentMetadataLink",
            "SPR.hasFilteredTPMLink",
            "SPR.hasMarkerGenesLink",
            "SPR.hasMatrixLink",
            "SPR.hasNormalisedCountsLink",
            "SPR.hasRawCountsLink",
            "SPR.hasResultsLink",
	};
	
	public static final String[] PR_ANNOTATION_PROPERTIES = new String[] {
			"PR.hasArrayExpressID",
			"PR.hasENAprojectID",
			"PR.hasDescription",
			"PR.hasGEOseriesID",
			"PR.hasINSDCprojectID",
			"PR.hasINSDCstudyID",
			"PR.hasInstitution",
			"PR.hasLoadDate",
			"PR.hasProjectID",
			"PR.hasPublicationLink",
			"PR.hasPublicationTitle",
			"PR.hasProjectRepositoryLink",
			"PR.hasUpdateDate"
	};
	
	public static final String[] SR_ANNOTATION_PROPERTIES = new String[] {
			"SR.hasSpecimenID",
			"SR.hasAssayID",
	};
	
	public static final String[] SPECIMEN_ANNOTATION_PROPERTIES = ArrayUtils.addAll(SPR_ANOTATION_PROPERTIES, SR_ANNOTATION_PROPERTIES);
	public static final String[] PROJECT_ANNOTATION_PROPERTIES = ArrayUtils.addAll(SPR_ANOTATION_PROPERTIES, PR_ANNOTATION_PROPERTIES);
	
	private void initializeInputStream(String inputFileName) {
		try {
			inputStream = new FileInputStream(inputFileName);
		} catch (FileNotFoundException e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		}
	}
	
	private void initializeOutputStream(String outputFileName) {
		try {
			outputStream = new FileOutputStream(outputFileName);
		} catch (FileNotFoundException e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		}
	}

	public MyModel(String NS, String inputFileName) {
		this.NS = NS;
		
		initializeInputStream(inputFileName);
		model = ModelFactory.createOntologyModel(PelletReasonerFactory.THE_SPEC);
		model.getReasoner().setParameter(ReasonerVocabulary.PROPsetRDFSLevel, ReasonerVocabulary.RDFS_FULL);

		model.read(inputStream, "RDF/XML");

		classMap = new HashMap<>();
		dataPropertyMap = new HashMap<>();
		objectPropertyMap = new HashMap<>();
		anotationPropertyMap = new HashMap<>();
		individualMap = new HashMap<>();
		
		classMap.put(null, null);
	}

	public void saveModel(String outputFileName) {
		initializeOutputStream(outputFileName);
		
		System.out.println("Saving model...");
		
		model.write(outputStream, "RDF/XML");
		
		System.out.println("Model saved.");
	}

	public Resource getOntClass(String className) {		
		Resource ontClass = classMap.get(className);

		if (!classMap.containsKey(className)) {
			ontClass = model.getResource(NS + className);
			classMap.put(className, ontClass);

			if (ontClass == null)
				System.err.println("Warning: Class " + className + " is not in the ontology model.");
			
		}
		return ontClass;
	}

	public Property getDataProperty(String propertyName) {
		Property property = dataPropertyMap.get(propertyName);

		if (!dataPropertyMap.containsKey(propertyName)) {
				property = model.getProperty(NS + propertyName);
				dataPropertyMap.put(propertyName, property);
				
				if (property == null) 
					System.err.println("Warning: Data property " + propertyName + " is not in the ontology model.");
		}
		return property;
	}
	
	public Property getObjectProperty(String propertyName) {
		Property property = objectPropertyMap.get(propertyName);

		if (!objectPropertyMap.containsKey(propertyName)) {
			property = model.getProperty(NS + propertyName);
			objectPropertyMap.put(propertyName, property);
			
			if (property == null) 
				System.err.println("Warning: Object property " + propertyName + " is not in the ontology model.");
		}
		return property;
	}
	
	private Property getAnotationProperty(String propertyName) {
		Property property = anotationPropertyMap.get(propertyName);

		if (!anotationPropertyMap.containsKey(propertyName)) {
			property = model.getProperty(NS + propertyName);
			anotationPropertyMap.put(propertyName, property);
			
			if (property == null) 
				System.err.println("Warning: Anotation property " + propertyName + " is not in the ontology model.");
		}
		return property;
	}
	
	public Resource getIndividual(String individualName) {
		Resource individual = individualMap.get(individualName);

		if (!individualMap.containsKey(individualName)) {
			
			if (!model.containsResource(ResourceFactory.createResource(NS + individualName))) 
				System.err.println("Warning: Individual " + individualName + " is not in the ontology model.");
			
			individual = model.getResource(NS + individualName);
					
			individualMap.put(individualName, individual);

		}
		return individual;
	}
	
	public void createSpecimen(String id) {
		Resource specimenClass = getOntClass(SPECIMEN_CLASS);
		Resource individual = model.createResource(NS + id, specimenClass);
		
		individualMap.put(id, individual);
	}
	
	public void createProject(String id) {
		Resource projectClass = getOntClass(PROJECT_CLASS);
		Resource individual = model.createResource(NS + id, projectClass);
		
		individualMap.put(id, individual);
	}
	
	public void addObjectPropertyToIndividual(String subject, String predicate, Object object) {
		if (object.toString().compareTo("null") == 0)
			return;
		
		Property property = getObjectProperty(predicate);
		Resource individual = individualMap.get(subject);
		Resource objectIndividual = getIndividual(object.toString());
		
		if (objectIndividual == null)
			return;
		
		model.add(individual, property, objectIndividual);
	}
	
	
	public void addDataPropertyToIndividual(String subject, String predicate, Object object) {
		if (object.toString().compareTo("null") == 0)
			return;

		Property property = getDataProperty(predicate);
		Resource individual = individualMap.get(subject);
		
		if (object instanceof String)
			model.add(individual, property, object.toString());
		else if (object instanceof Double)
			model.addLiteral(individual, property, (double) object);
		else if (object instanceof Integer)
			model.addLiteral(individual, property, (int) object);
		else if (object instanceof Boolean)
			model.addLiteral(individual, property, (boolean) object);

	}
	
	public void addAnotationPropertyToIndividual(String subject, String predicate, Object object) {
		if (object.toString().compareTo("null") == 0)
			return;
 
		Property property = getAnotationProperty(predicate);
		
		Resource individual = getIndividual(subject);
		
		if (object instanceof String)
			model.add(individual, property, object.toString());
		else if (object instanceof Double)
			model.addLiteral(individual, property, (double) object);
		else if (object instanceof Integer)
			model.addLiteral(individual, property, (int) object);
		else if (object instanceof Boolean)
			model.addLiteral(individual, property, (boolean) object);
		
	}


	@SuppressWarnings("resource")
	public boolean validateModel() {
		FileWriter writer = null;
		
		try {
			writer = new FileWriter(validateLogFileName);
		} catch (IOException e) {
			e.printStackTrace();
			return false;
		}
		
		System.out.println("Validating model with " + model.getReasoner().getClass().getSimpleName() + "...");
				
		ValidityReport validity = model.validate();
		if (validity.isValid()) {
			System.out.println("Model validated without errors.");
			return true;
		}
		
		System.out.println("Conflicts have occured during validation:");
		
		for (Iterator<Report> i = validity.getReports(); i.hasNext();) {
			ValidityReport.Report report = (ValidityReport.Report) i.next();
			
			try {
				writer.write(report.toString());
			} catch (IOException e) {
				e.printStackTrace();
				return false;
			}
			
			System.out.println(report);
		}
		
		return false;
	}
	
	public Model getModel() {
		return model;
	}


}
