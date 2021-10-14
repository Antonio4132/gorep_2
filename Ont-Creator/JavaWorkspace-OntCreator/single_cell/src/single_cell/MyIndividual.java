package single_cell;

import java.util.List;

import org.json.JSONArray;
import org.json.JSONObject;

public abstract class MyIndividual {

	private String id;
	
	private JSONObject jsonIndividual;
	
	private MyModel model;

	private String readStringFromJson(JSONObject jsonObject, String field) {
		String string = null;

		try {
			string = jsonObject.getString(field);
		} catch (Exception e) {
			return null;
		}

		return string;
	}

	public MyIndividual(JSONObject jsonIndividual, MyModel model) {
		initIndividual(jsonIndividual);

		this.model = model;
	}

	private void initIndividual(JSONObject jsonIndividual) {

		this.id = readStringFromJson(jsonIndividual, "ID");
		
		this.jsonIndividual = jsonIndividual;
		
	}
	
	public Boolean addToModel() {
		
		createIndividual(id);
				
		JSONObject objectProperties = jsonIndividual.getJSONObject("ObjectProperties");
		// Add object properties
		for (String propertyName : getObjectProperties()) {
			try {
				Object propertieValueObject = objectProperties.get(propertyName);
				
				if (propertieValueObject instanceof JSONArray) {
					List<Object> list = objectProperties.getJSONArray(propertyName).toList();
					
					for (Object propertyValue : list) {
						if (propertyValue instanceof Object)
							model.addObjectPropertyToIndividual(id, propertyName, propertyValue);
					}
				} else if (propertieValueObject instanceof Object){
					model.addObjectPropertyToIndividual(id, propertyName, propertieValueObject);
				}
			} catch (Exception e) {
				e.printStackTrace();
				System.exit(1);
			}

		}
		
		JSONObject dataProperties = jsonIndividual.getJSONObject("DataProperties");
		// Add data properties
		for (String propertyName : getDataProperties()) {
			try {
				Object propertieValueObject = dataProperties.get(propertyName);
				
				if (propertieValueObject instanceof JSONArray) {
					List<Object> list = dataProperties.getJSONArray(propertyName).toList();
					
					for (Object propertyValue : list) {
						if (propertyValue instanceof Object)
							model.addDataPropertyToIndividual(id, propertyName, propertyValue);
					}
				} else if (propertieValueObject instanceof Object){
					model.addDataPropertyToIndividual(id, propertyName, propertieValueObject);
				}
			} catch (Exception e) {
				e.printStackTrace();
				System.exit(1);
			}

		}
		
		JSONObject anotationProperties = getJsonIndividual().getJSONObject("AnnotationProperties");
		// Add anotation properties
		for (String propertyName : getAnnotationProperties()) {
			try {
				Object propertieValueObject = anotationProperties.get(propertyName);
				
				if (propertieValueObject instanceof JSONArray) {
					List<Object> list = anotationProperties.getJSONArray(propertyName).toList();
					
					for (Object propertyValue : list) {
						if (propertyValue instanceof Object)
							getModel().addAnotationPropertyToIndividual(getId(), propertyName, propertyValue);
					}
				} else if (propertieValueObject instanceof Object){
					getModel().addAnotationPropertyToIndividual(getId(), propertyName, propertieValueObject);
				}
			} catch (Exception e) {
				e.printStackTrace();
				System.exit(1);
			}

		}
		
		return true;
	}
	

	protected abstract String[] getAnnotationProperties();

	public MyModel getModel() {
		return model;
	}
	
	public JSONObject getJsonIndividual() {
		return jsonIndividual;
	}
	
	public String getId() {
		return id;
	}
	
	protected abstract void createIndividual(String id);

	protected abstract String[] getObjectProperties();
	
	protected abstract String[] getDataProperties();

}
