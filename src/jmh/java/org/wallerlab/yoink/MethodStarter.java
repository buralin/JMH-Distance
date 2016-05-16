package org.wallerlab.yoink;

import static org.wallerlab.yoink.domain.Region.BUFFERZONE;
import static org.wallerlab.yoink.domain.Region.MM;
import static org.wallerlab.yoink.domain.Region.QMZONE;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.stream.Collectors;

import org.wallerlab.yoink.domain.GridPoint;
import org.wallerlab.yoink.domain.GridPointInterface;
import org.wallerlab.yoink.domain.Molecule;
import org.wallerlab.yoink.domain.Point;
import org.wallerlab.yoink.domain.Region;
import org.wallerlab.yoink.service.Commons;
import org.wallerlab.yoink.service.DistanceCalculator;
import org.wallerlab.yoink.service.DistanceKernel;
import org.wallerlab.yoink.service.ICalculateGridDistance;
import org.wallerlab.yoink.service.IDistance;
import org.wallerlab.yoink.service.JBLAS;
import org.wallerlab.yoink.service.JBLASDistGrid;
import org.wallerlab.yoink.service.JavaDistanceSplit;
import org.wallerlab.yoink.service.ShortestDistanceJava;
import org.wallerlab.yoink.service.ShortestDistanceKerlenAllocationtTme;
import org.wallerlab.yoink.service.ShortestDistanceKernel;


public class MethodStarter {
	public float[] calculateDistanceKernel(List<Molecule> molecules,Point point )
	{
		IDistance dc = new DistanceKernel();
		float[] distances =  dc.calculateDistance(molecules, point);
		//pause();
		return distances; 	
	}
	public float[] calculateDistanceJava(List<Molecule> molecules,Point point )
	{
		IDistance dc = new DistanceCalculator();
		return dc.calculateDistance(molecules, point);
    }
	public float [] calculateDistanceCommons(List<Molecule> molecules,Point point )
	{
		IDistance dc = new Commons();
		float[] distances = dc.calculateDistance(molecules, point);
		//pause();
		return distances;

	}
	public float [] calculateDistanceJBlas(List<Molecule> molecules,Point point )
	{
		IDistance dc = new JBLAS();
		return dc.calculateDistance(molecules, point);
	}
	public float [] calculateDistanceJavaSplit(List<Molecule> molecules,Point point )
	{
		IDistance dc = new JavaDistanceSplit();
		return dc.calculateDistance(molecules, point);
	}
	public float [] calculateGridDistance (List<Molecule> molecules, GridPoint grid)
	{
		ICalculateGridDistance dc = new ShortestDistanceJava();
		return dc.calculateDistance(molecules, grid);
	}
	public float [] calculateGridDistanceKernel (List<Molecule> molecules, GridPoint grid)
	{
		ICalculateGridDistance dc = new ShortestDistanceKernel();
		return dc.calculateDistance(molecules, grid);
	}
	public float [] calculateGridDistanceJBLAS (List<Molecule> molecules, GridPoint grid)
	{
		ICalculateGridDistance dc = new JBLASDistGrid();
		return dc.calculateDistance(molecules, grid);
	}
	public float [] calculateGridDistanceKernelAllocationTime (List<Molecule> molecules, GridPoint grid)
	{
		ICalculateGridDistance dc = new ShortestDistanceKerlenAllocationtTme();
		return dc.calculateDistance(molecules, grid);
	}
	
	
	
}
