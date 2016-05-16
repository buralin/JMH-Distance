package org.wallerlab.yoink.service;

import java.util.List;

import org.apache.commons.math3.geometry.euclidean.threed.Vector3D;
import org.wallerlab.yoink.domain.Atom;
import org.wallerlab.yoink.domain.Molecule;
import org.wallerlab.yoink.domain.Point;

public class Commons implements IDistance
{
	
	public float[] calculateDistance(List<Molecule> molecules,Point com ) 
	{
		double [] comVector = {com.getX(),com.getY(), com.getZ()};
		
		int atomIndex =0;
		double dist = 0;
		int counter=0;
		float[] distarray = new float [molecules.size()*10];	
		
		for(int i = 0; i<molecules.size();i++){
		    for(atomIndex=0; atomIndex < molecules.get(i).getAtoms().size();atomIndex ++){
		    	double [] in1 = new double [3];
		    	in1[0] = molecules.get(i).getAtoms().get(atomIndex).getX();
		    	in1[1] = molecules.get(i).getAtoms().get(atomIndex).getY();
		    	in1[2] = molecules.get(i).getAtoms().get(atomIndex).getZ();
		        dist = commonsdist (in1, comVector);
		        distarray[counter] = (float)dist; 
		        counter ++;
		    } 
		}
		return distarray;
	}
	
	public double commonsdist (double[] in1, double[] in2)
	{
		Vector3D ina = new Vector3D(in1);
		Vector3D inb = new Vector3D(in2);
		return ina.distance(inb);
 }

}