package org.wallerlab.yoink.domain;
import java.util.Random;

public class GridPoint implements GridPointInterface
{
	double points [] = new double [3000]; 
	
	public GridPoint() {
		this.points = makingPoints();
	}
	public double [] makingPoints (){
		double [] a = new double [3000];
		
		 for (int i = 0; i<3000;i++)
		  {
			Random random = new Random();
			a[i] = random.nextDouble()*10;
		  }
		return a;
	}
	public double[] getPoints()
	{
		return points;
	}
	public void setPoints(double[] points)
	{
		this.points = points;
	}

}
