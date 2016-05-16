package org.wallerlab.yoink;

import java.util.List;
import java.util.stream.Collectors;
import java.util.stream.IntStream;

import org.openjdk.jmh.annotations.Benchmark;
import org.openjdk.jmh.annotations.BenchmarkMode;
import org.openjdk.jmh.annotations.Measurement;
import org.openjdk.jmh.annotations.Mode;
import org.openjdk.jmh.annotations.Scope;
import org.openjdk.jmh.annotations.Setup;
import org.openjdk.jmh.annotations.State;
import org.openjdk.jmh.annotations.Warmup;
import org.openjdk.jmh.infra.Blackhole;
import org.openjdk.jmh.runner.Runner;
import org.openjdk.jmh.runner.options.Options;
import org.openjdk.jmh.runner.options.OptionsBuilder;
import org.wallerlab.yoink.domain.GridPoint;
import org.wallerlab.yoink.domain.Molecule;
import org.wallerlab.yoink.domain.Point;


@State(Scope.Thread)
public class SetOfBenchmarks {

	double qmThreshold;
	double bufferThreshold;
	List<Molecule> molecules;
	Molecule core = new Molecule(0);
	MethodStarter starter;
	Point point;
	GridPoint grid;

	@Setup
	public void setup() {
		qmThreshold = 2.0;
		bufferThreshold = 5.0;
		System.out.println("setting up molecules");
		
	
		molecules = 
				IntStream.range(0,1000)
						 .parallel()
						 .mapToObj(i -> new Molecule(i))
						 .collect(Collectors.toList());
		point = new Point(0.0,0.0,0.0);
		System.out.println("Qm threshold" + qmThreshold);
		System.out.println("buffer threshold" + bufferThreshold);
		System.out.println("molecules" + molecules.size());
		starter = new MethodStarter();

	}
/*
	@Benchmark
	public void commons(Blackhole bh) {
		bh.consume(starter.calculateDistanceCommons( molecules,point));
	}*/
	@Benchmark
	@Warmup(iterations = 1, time = 1)
	@Measurement(iterations = 1, time = 3)
	@BenchmarkMode(Mode.SingleShotTime)
	public void distanceKernelSharedMemory2(Blackhole bh) {
		bh.consume(starter.calculateDistanceSharedMemorySplit2( molecules,point));
	}
	
	@Benchmark
	@Warmup(iterations = 1, time = 1)
	@Measurement(iterations = 1, time = 3)
	@BenchmarkMode(Mode.SingleShotTime)
	public void distanceKernelSharedMemory(Blackhole bh) {
		bh.consume(starter.calculateDistanceKernelShared( molecules,point));
	}
	@Benchmark
	@Warmup(iterations = 1, time = 1)
	@Measurement(iterations = 1, time = 3)
	@BenchmarkMode(Mode.SingleShotTime)
	public void distanceKernel(Blackhole bh) {
		bh.consume(starter.calculateDistanceKernel( molecules,point));
	}
    @Benchmark
    @Warmup(iterations = 1, time = 1)
    @Measurement(iterations = 1, time = 3)
    @BenchmarkMode(Mode.SingleShotTime)
	public void distancejava (Blackhole bh){
		bh.consume(starter.calculateDistanceJava( molecules,point));
	}
    @Benchmark
    @Warmup(iterations = 1, time = 1)
    @Measurement(iterations = 1, time = 3)
    @BenchmarkMode(Mode.SingleShotTime)
	public void distanceCommons (Blackhole bh){
		bh.consume(starter.calculateDistanceCommons( molecules,point));
	}
    @Benchmark
    @Warmup(iterations = 1, time = 1)
    @Measurement(iterations = 1, time = 3)
    @BenchmarkMode(Mode.SingleShotTime)
	public void distanceJBlas (Blackhole bh){
		bh.consume(starter.calculateDistanceJBlas( molecules,point));
	}
    @Benchmark
    @Warmup(iterations = 1, time = 1)
    @Measurement(iterations = 1, time = 3)
    @BenchmarkMode(Mode.SingleShotTime)
	public void distanceJavaSplit (Blackhole bh){
		bh.consume(starter.calculateDistanceJBlas( molecules,point));
	}
	/*@Benchmark
    @Warmup(iterations = 1, time = 1)
    @Measurement(iterations = 1, time = 3)
    @BenchmarkMode(Mode.SingleShotTime)
	public void distanceGridJava (Blackhole bh){
		bh.consume(starter.calculateGridDistance( molecules,grid));
	}
	
	@Benchmark
    @Warmup(iterations = 1, time = 1)
    @Measurement(iterations = 1, time = 3)
    @BenchmarkMode(Mode.SingleShotTime)
	public void  calculateGridDistanceKernel(Blackhole bh){
		bh.consume(starter.calculateGridDistanceKernel( molecules,grid));
	}
	@Benchmark
    @Warmup(iterations = 1, time = 1)
    @Measurement(iterations = 1, time = 3)
    @BenchmarkMode(Mode.SingleShotTime)
	public void  calculateGridDistanceKernelShared(Blackhole bh){
		bh.consume(starter.calculateGridDistanceKernelShared( molecules,grid));
	}*/
	
    public static void main(String[] args) throws Exception {
		Options options = new OptionsBuilder()
				.include(SetOfBenchmarks.class.getSimpleName())
				.warmupIterations(1)
				.measurementIterations(1)
				.forks(1)
				.build();
		new Runner(options).run();
	}
}
