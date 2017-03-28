% WRITE_IMGS - function to generate responseimages 
%            - after convolving with log-gabor filters
%
function write_imgs(image_path, output_dir, scales, orientations)
  
  minWaveLength = 3;
  mult = 2;
  sigmaOnf = 0.65;
  dThetaOnSigma = 1.3;

  image = imread(image_path);
  [E0, _] = gaborconvolve(image, scales, orientations, 
                              minWaveLength, mult, sigmaOnf, dThetaOnSigma);

  for s = 1:scales
    for o = 1:orientations
      or = (o-1)*pi/orientations;
      fprintf("Writing image for scale: %d, orientation: %f\n",s,or);
      out_path = [output_dir '/s-' int2str(s) 'o-' ...
                            int2str(o) '|' int2str(orientations) '.png'];
      imwrite(real(E0{s,o}),out_path);
    end
  end