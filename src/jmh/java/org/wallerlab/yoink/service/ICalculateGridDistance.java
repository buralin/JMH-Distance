package org.wallerlab.yoink.service;

import java.util.List;

import org.wallerlab.yoink.domain.GridPoint;
import org.wallerlab.yoink.domain.GridPointInterface;
import org.wallerlab.yoink.domain.Molecule;
import org.wallerlab.yoink.domain.Point;

public interface ICalculateGridDistance 
{
	 float[] calculateDistance(List<Molecule> molecules, GridPoint grid);
}
