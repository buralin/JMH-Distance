package org.wallerlab.yoink.service;

import java.util.List;
import java.util.Random;

import org.apache.commons.math3.geometry.euclidean.threed.Vector3D;
import org.wallerlab.yoink.domain.GridPoint;
import org.wallerlab.yoink.domain.Molecule;

public class JBLASDistGrid implements ICalculateGridDistance{
	
	public float[] calculateDistance(List<Molecule> molecules, GridPoint grid) 
	{
		
		double [] points = makingPoints();
		
		
		int atomIndex =0;
		int count = 0;
		double dist = 0;
		float [] distarray = new float [molecules.get(0).getAtoms().size()*10000*2000];//80000 Molcecules, 20000 GridPoints
		
		for(int i = 0; i<molecules.size();i++)
		{
		    for(atomIndex=0; atomIndex < molecules.get(i).getAtoms().size();atomIndex ++)
		    {
			        double in1 [] = new double [3];
		    	in1[0] = molecules.get(i).getAtoms().get(atomIndex).getX();
		    	in1[1] = molecules.get(i).getAtoms().get(atomIndex).getY();
		    	in1[2] = molecules.get(i).getAtoms().get(atomIndex).getZ();
		    	for ( int j = 0 ; j < points.length; j +=3)
		    	{
		    	double in2 [] = new double [3];
		    	in2[0] = points[j];
		    	in2[1] = points[j+1];
		    	in2[2] = points[j+2];
		        dist = javadist (in1, in2);
		        distarray[count] = (float)dist;
		        count += 1;
		        } 
		     }
	    }
	return distarray; 
	}
	public double javadist (double[] in1, double[] in2)
	{
		Vector3D in13D = new Vector3D(in1);
		Vector3D in23D = new Vector3D(in2);
		return in13D.distance(in23D);
    }
	public double [] makingPoints (){
		double [] a = new double [6000];
		
		 for (int i = 0; i<6000;i++)
		  {
			Random random = new Random();
			a[i] = random.nextDouble()*10;
		  }
		return a;
	}
}

