//importamos todas las librerías que vamos a ocupar , openCV está en C:
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/opencv.hpp>
#include <iostream>
#include <math.h>

using namespace cv;
using namespace std;
//Funcion que nos permite obtener la imagen que viene en el folder y para identificar si tiene datos o no
Mat obtenerImagen() {
	//la imagen se encuentra en la carpeta donde esta el source.cpp
	char NombreImagen[] = "lena.jpg";
	//creamos la matriz en donde vamos a alojar nuestra imagen inicial
	Mat imagen;
	//mandamos a leer la imagen y la almacenamos en la variable previamente creada llamada "imagen"
	imagen = imread(NombreImagen, IMREAD_UNCHANGED);
	//en dado caso de que no se haya detectado la imagen, se acabará el programa  ya que no hay datos
	if (!imagen.data)
	{
		cout << "Error al cargar la imagen: " << NombreImagen << endl;
		exit(1);
	}
	//devolvemos la imagen una vez que nos hayamos asegurado que si existe
	return imagen;
}
//Funcion que nos sirve para obtener la matriz/kernel gaussiana para obetener el filtro
Mat gauss(int filasColumnas, float sigma) {
	//dependiendo del valor que agrege el usuario, crearemos una variable llamada "border"
	//con la cual nos servirá para saber de cuando será nuestro borde de la imagen completa
	int bordesImagen = (filasColumnas - 1) / 2;
	//inicializaremos la variable Y, la cual será el primer valor de nuestra matriz (x,y) en posición Y
	//es decir, si el valor ingresado es 5, nuestro borde es 2, por lo que el primer valor de nuestra matriz Y
	// será 2
	int variableY = bordesImagen;
	//creamos las variables en donde guardaremos la matriz en el punto X y la matriz en el punto Y
	//y la matriz gaussiana
	Mat PosicionX(filasColumnas, filasColumnas, CV_32SC1);
	Mat PosicionY(filasColumnas, filasColumnas, CV_32SC1);
	Mat KernelGauss(filasColumnas, filasColumnas, CV_32FC1);
	//con estos dos for recorreremos todas nuestras casillas de las matrices x y y, y vamos a incrementar o 
	//disminuir en cada casilla dependiendo si es filas o columnas, la variable X dependerá de las 
	//columnas, mientras que la Y de las filas, por ejemplo, si el usuario ingresa 5, nuestra variable Y 
	//será 2, mientras que la X es la negativa de Y, por lo que nuestra primera casilla será (-2,2)
	//dicho esto, al final de las iteraciones con el ejemplo de 5 nos debería quedar así 
	//(-2,2) (-1,2) (0,2) (1,2) (2,2)
	//(-2,1) (-1,1) (0,1) (1,1) (2,1)
	//(-2,0) (-1,0) (0,0) (1,0) (2,0)
	//(-2,-1) (-1,-1) (0,-1) (1,-1) (2,-1)
	//(-2,-2) (-1,-2) (0,-2) (1,-2) (2,-2)
	//cabe mencionar que se usaron 2 matrices para guardar la posición de X y Y respectivamente
	for (int i = 0; i < filasColumnas; i++) { //filas
		int variableX = -( bordesImagen);
		for (int j = 0; j < filasColumnas; j++) { //columnas
			PosicionX.at<int>(i, j) = variableX;
			PosicionY.at<int>(i, j) = variableY;
			variableX++;
		}
		variableY--;
	}
	//una vez obtenidas las matriz X y Y, seguimos la formula de g(x,y) para obtener el valor de nuestro kernel
	//para obtener la matriz gaussiana
	for (int i = 0; i < filasColumnas; i++) { //filas
		for (int j = 0; j < filasColumnas; j++) { //columnas
			KernelGauss.at<float>(i, j) = (1 / (2 * CV_PI * pow(sigma, 2))) * exp(-1 * (pow(PosicionX.at<int>(i, j), 2) + pow(PosicionY.at<int>(i, j), 2)) / (2 * pow(sigma, 2)));
		}
	}
	//los siguientes 4 for nos sirven para normalizar nuestra matriz gaussiana, esto lo logramos obteniendo
	//el promedio de todos los valores y dividiendo cada valor sobre el promedio
	double gaussianoSuma = 0;
	for (int i = 0; i < filasColumnas; i++) {
		for (int j = 0; j < filasColumnas; j++) {
			gaussianoSuma = KernelGauss.at<float>(i, j) + gaussianoSuma;
		}
	}

	for (int i = 0; i < filasColumnas; i++) {
		for (int j = 0; j < filasColumnas; j++) {
			KernelGauss.at<float>(i, j) = KernelGauss.at<float>(i, j) / gaussianoSuma;
		}
	}

	return KernelGauss;
}
//Nuestra siguiente funcion lo que hace es agregar los bordes a la imagen original
Mat imagenBordes(Mat gris, int filasColumnas) {
	//creamos nuestra matriz de destino en donde contendrá la imagen mas los bordes 
	Mat MatrizDestinoBordes(filasColumnas - 1 + gris.rows, filasColumnas - 1 + gris.cols, CV_8UC1, Scalar(0));
	//obtenemos el valor de los bordes como ya fue explicado en una función anterior 
	int bordesImagen = (filasColumnas - 1) / 2;
	//pegamos la imagen en gris en medio de nuestra imagen de destino para así lograr los bordes
	for (int i = 0; i < gris.rows; i++) {
		for (int j = 0; j < gris.cols; j++) {
			MatrizDestinoBordes.at<uchar>(bordesImagen + i, bordesImagen + j) = gris.at<uchar>(i, j);
		}
	}
	//devolvemos la imagen en gris con los bordes
	return MatrizDestinoBordes;
}
//la siguiente función solo nos servira para imprimir en consola todos los valores solicitados y 
//las imagenes
void imprimir(Mat destino, Mat imagen, Mat filtro, Mat sobel, Mat sobelAngulo, Mat ImagenUMS, Mat ImagenNuevaUmbral) {
	namedWindow("Con bordes", WINDOW_AUTOSIZE);
	imshow("Con bordes", destino);
	namedWindow("Original", WINDOW_AUTOSIZE);
	imshow("Original", imagen);
	namedWindow("filtro", WINDOW_AUTOSIZE);
	imshow("filtro", filtro);
	namedWindow("sobel", WINDOW_AUTOSIZE);
	imshow("sobel", sobel);
	namedWindow("Angulo", WINDOW_AUTOSIZE);
	imshow("Angulo", sobelAngulo);
	namedWindow("UMS", WINDOW_AUTOSIZE);
	imshow("UMS", ImagenUMS);
	namedWindow("Canny", WINDOW_AUTOSIZE);
	imshow("Canny", ImagenNuevaUmbral);

	cout << "\nImagen.rows[" << imagen.rows << "] Imagen.cols[" << imagen.cols << "]" << endl;
	cout << "Destino.rows[" << destino.rows << "] Destino.cols[" << destino.cols << "]" << endl;
	cout << "Filtro.rows[" << filtro.rows << "] Filtro.cols[" << filtro.cols << "]" << endl;
	cout << "sobel.rows[" << sobel.rows << "] sobel.cols[" << sobel.cols << "]" << endl;
	cout << "Angulo.rows[" << sobelAngulo.rows << "] Angulo.cols[" << sobelAngulo.cols << "]" << endl;
	cout << "UMS.rows[" << ImagenUMS.rows << "] UMS.cols[" << ImagenUMS.cols << "]" << endl;
	cout << "Canny.rows[" << ImagenNuevaUmbral.rows << "] Canny.cols[" << ImagenNuevaUmbral.cols << "]" << endl;
}
//Esta función realiza el filtro utilizando nuestro kernel (matriz gaussiana) y la imagen con bordes 
void filtroGaussiano(Mat gauss, Mat destino, int filasColumnas, Mat& ImagenConfiltro) {
	//creamos una variable del tamaño que ingresó el usuario para la realizar el producto punto de la
	//matriz 
	Mat productoPuntoMatriz(filasColumnas, filasColumnas, CV_32FC1);
	//creamos una variable temporal la cual nos servirá para guardar valores momentaneament, mas adelante 
	//será mejor explicada esta variable
	float casillaTemporal=0;
	//con estos dos for vamos a recorrer nuestra matriz donde estará guardado el filtro gaussiano 
	for (int i = 0; i < ImagenConfiltro.rows; i++) {
		for (int j = 0; j < ImagenConfiltro.cols; j++) {
			//esta función lo que hace es señalar una parte mas pequeña de la matriz de tamaño que ingresó el
			//usuario de nuestra imagen con bordes, esto para realizar de una manera más sencilla 
			//nuestro producto punto
			productoPuntoMatriz = Mat(destino, Rect(j, i, filasColumnas, filasColumnas));
			//cada casilla el valor de Vtemp se irá reiniciando para guardar el nuevo valor de la casilla
			casillaTemporal = 0;
			//en cada casilla de origen vamos a ir a realizar el producto punto con centro (i,j) tomando
			//sus alrededores dependiendo del valor que ingresó el usuario del borde 
			for (int i = 0; i < filasColumnas; i++) {
				for (int j = 0; j < filasColumnas; j++) {
					//aquí se va realizando el producto punto y se va guardando la suma en vtemp
					casillaTemporal = casillaTemporal + (gauss.at<float>(i, j) * productoPuntoMatriz.at<uchar>(i, j));

				}
			}
			//una vez que tenemos el producto punto de (i,j) y sus alrededores, vamos guardando el valor 
			//de esa nueva casilla en nuestra matriz filtro
			ImagenConfiltro.at<uchar>(i, j) = casillaTemporal;

		}
	}
	//no retornamos ningún valor porque se trabajó con referencias esta función
}
//esta funcion nos sirve para obtener el producto sobel, solo ocupamos el filtro que previamente realizamos
Mat operadorSobel(Mat imagenFiltro) {
	//creamos las variables en donde guardaremos los kernel de Gx y Gy
	Mat matrizGxKernel(3, 3, CV_32FC1);
	Mat matrizGyKernel(3, 3, CV_32FC1);
	//Se ingresa manualmente el valor del kernel que proponen para realizar este método
	matrizGxKernel.at<float>(0, 0) = -1;
	matrizGxKernel.at<float>(0, 1) = 0;
	matrizGxKernel.at<float>(0, 2) = 1;
	matrizGxKernel.at<float>(1, 0) = -2;
	matrizGxKernel.at<float>(1, 1) = 0;
	matrizGxKernel.at<float>(1, 2) = 2;
	matrizGxKernel.at<float>(2, 0) = -1;
	matrizGxKernel.at<float>(2, 1) = 0;
	matrizGxKernel.at<float>(2, 2) = 1;

	matrizGyKernel.at<float>(0, 0) = -1;
	matrizGyKernel.at<float>(0, 1) = -2;
	matrizGyKernel.at<float>(0, 2) = -1;
	matrizGyKernel.at<float>(1, 0) = 0;
	matrizGyKernel.at<float>(1, 1) = 0;
	matrizGyKernel.at<float>(1, 2) = 0;
	matrizGyKernel.at<float>(2, 0) = 1;
	matrizGyKernel.at<float>(2, 1) = 2;
	matrizGyKernel.at<float>(2, 2) = 1;
	//creamos dos valores temporales que nos servirá para guardar la casilla de los productos puntos de 
	//nuestros kernel de Gx y GY con el filtro, para después guardarlos en otras matrices 
	float temporalX = 0, temporalY = 0;
	//creamos nuestra matriz de 3x3 la cual nos servirá para realizar el producto punto de Gx y Gy
	//por lo que también debe ser de 3x3
	Mat temporalProductoPunto(3, 3, CV_8UC1);
	//creamos nuestras matrices en donde guardaremos el valor de sobelX, sobelY, sobel y para el ángulo 
	Mat sobelX(imagenFiltro.rows, imagenFiltro.cols, CV_32FC1, Scalar(0));
	Mat sobelY(imagenFiltro.rows, imagenFiltro.cols, CV_32FC1, Scalar(0));
	Mat sobel(imagenFiltro.rows, imagenFiltro.cols, CV_8UC1, Scalar(0));
	Mat sobelAngulo(imagenFiltro.rows, imagenFiltro.cols, CV_32FC1, Scalar(0));
	//al igual que la función para obtener el filtro gaussiano, aquí tambien ocuparemos 4 for, los 2 primeros
	//para recorrer las casillas de nuestra imagen con el filtro y los otros 2 for para realizar el 
	//producto punto de cada matriz 3x3 dependiendo del punto (i,j)
	for (int i = 2; i < imagenFiltro.rows-2; i++) {
		for (int j = 2; j < imagenFiltro.cols-2; j++) {
			//cada que va avanzando en (i,j) se guardará una matriz de 3x3 con adyacencia en (i,j)
			//esto para realizar el producto punto
			temporalProductoPunto = Mat(imagenFiltro, Rect(j, i, 3, 3));
			//cada iteracion de casilla reiniciaremos el valor de vtemp y vtemp2 en donde se guardará el producto
			//punto, en Vtemp se guardará el producto punto de la casilla con respecto a (i,j) y Gx, mientras 
			//que vtemp2 guardará para Gy
			temporalX = 0;
			temporalY = temporalX;
			//como se mencionó, estos dos for son para realizar el producto punto, mismo método que se ocupó
			//para la matriz gaussiana
			for (int i = 0; i < 3; i++) {
				for (int j = 0; j < 3; j++) {

					temporalX = temporalX + (matrizGxKernel.at<float>(i, j) * temporalProductoPunto.at<uchar>(i, j));
					temporalY = temporalY + (matrizGyKernel.at<float>(i, j) * temporalProductoPunto.at<uchar>(i, j));
				}
			}
			//por ultimo, en cada producto punto se va guardando en sobelX y sobelY respectivamente
			sobelX.at<float>(i, j) = temporalX;
			sobelY.at<float>(i, j) = temporalY;
		}
	}
	//imprimimos las imagenes de sobelX y sobelY 
	namedWindow("sobelX", WINDOW_AUTOSIZE);
	imshow("sobelX", sobelX);
	namedWindow("sobelY", WINDOW_AUTOSIZE);
	imshow("sobelY", sobelY);
	//con estos dos for calculamos la magnitud de G y lo vamos guardando directamente en sobel
	for (int i = 0; i < sobel.cols; i++) {
		for (int j = 0; j < sobel.cols; j++) {
			sobel.at<uchar>(i, j) = sqrt(pow((sobelX.at<float>(i, j)),2) + pow((sobelY.at<float>(i, j)),2));
		}
	}
	
	//Creamos el ángulo de cada casilla dependiendo con ayuda del arco tangente y sobelX con sobelY 	
	for (int i = 0; i < imagenFiltro.rows; i++) {
		for (int j = 0; j < imagenFiltro.cols; j++) {
			float Y = sobelY.at<float>(i, j);
			float X = sobelX.at<float>(i, j);
			//se ocupará la función atan2 de math.h
			sobelAngulo.at<float>(i, j) = atan2(Y,X);
		}
	}
	//por último devolvemos la imagen con producto Sobel
	return sobel;
}

Mat angulo(Mat imagenFiltro) {
	//	ADVERTENCIA AL LEECTOR
	//
	//esta función es exactamente la misma que "operadorSobel", esto se debe a que c++ no puede retornar 2
	//valores y me di cuenta demasiado tarde, por lo que en vez de crear 4 funciones mas para crear individualmente
	//sobelX y sobelY, se realizó sobel y sobel angulo en la misma función, solo que retorno valores distintos

	Mat matrizGxKernel(3, 3, CV_32FC1);
	Mat matrizGyKernel(3, 3, CV_32FC1);

	matrizGxKernel.at<float>(0, 0) = -1;
	matrizGxKernel.at<float>(0, 1) = 0;
	matrizGxKernel.at<float>(0, 2) = 1;
	matrizGxKernel.at<float>(1, 0) = -2;
	matrizGxKernel.at<float>(1, 1) = 0;
	matrizGxKernel.at<float>(1, 2) = 2;
	matrizGxKernel.at<float>(2, 0) = -1;
	matrizGxKernel.at<float>(2, 1) = 0;
	matrizGxKernel.at<float>(2, 2) = 1;

	matrizGyKernel.at<float>(0, 0) = -1;
	matrizGyKernel.at<float>(0, 1) = -2;
	matrizGyKernel.at<float>(0, 2) = -1;
	matrizGyKernel.at<float>(1, 0) = 0;
	matrizGyKernel.at<float>(1, 1) = 0;
	matrizGyKernel.at<float>(1, 2) = 0;
	matrizGyKernel.at<float>(2, 0) = 1;
	matrizGyKernel.at<float>(2, 1) = 2;
	matrizGyKernel.at<float>(2, 2) = 1;

	float temporalX=0, temporalY=0;

	Mat temporalProductoPunto(3, 3, CV_8UC1);
	Mat sobelX(imagenFiltro.rows, imagenFiltro.cols, CV_32FC1, Scalar(0));
	Mat sobelY(imagenFiltro.rows, imagenFiltro.cols, CV_32FC1, Scalar(0));
	Mat sobel(imagenFiltro.rows, imagenFiltro.cols, CV_8UC1, Scalar(0));
	Mat sobelAngulo(imagenFiltro.rows, imagenFiltro.cols, CV_32FC1, Scalar(0));

	for (int i = 2; i < imagenFiltro.rows - 2; i++) {
		for (int j = 2; j < imagenFiltro.cols - 2; j++) {
			temporalProductoPunto = Mat(imagenFiltro, Rect(j, i, 3, 3));
			temporalX = 0;
			temporalY = temporalX;
			for (int i = 0; i < 3; i++) {
				for (int j = 0; j < 3; j++) {
					temporalX = temporalX + (matrizGxKernel.at<float>(i, j) * temporalProductoPunto.at<uchar>(i, j));
					temporalY = temporalY + (matrizGyKernel.at<float>(i, j) * temporalProductoPunto.at<uchar>(i, j));
				}
			}
			sobelX.at<float>(i, j) = temporalX;
			sobelY.at<float>(i, j) = temporalY;
		}
	}

	//angulo	
	for (int i = 0; i < imagenFiltro.rows; i++) {
		for (int j = 0; j < imagenFiltro.cols; j++) {
			float Y = sobelY.at<float>(i, j);
			float X = sobelX.at<float>(i, j);
			sobelAngulo.at<float>(i, j) = atan2(Y, X);
		}
	}
		//como se mención, es igual que la función anterior pero retorno el angulo en lugar del sobel
	return sobelAngulo;
}
//como primer paso para crear el método "canny", se debe crear la "Non-Maximum Suppression"
Mat UMS(Mat angulo, Mat sobel) {
	//como primer paso creamos una matriz del tamaño de nuestro angulo, esto se debe a que 
	//lo necesita para crearlo
	Mat imagenUMS(angulo.rows, angulo.cols, CV_8UC1, Scalar(0));
	//creamos 2 for para ir recorriendo la imagen en donde se guardó el angulo del método sobel
	for (int i = 0; i < angulo.rows; i++) {
		for (int j = 0; j < angulo.cols; j++) {
			//creamos dos variables con el valor máximo que puede tener un pixel en una foto 
			float x=255,r=x;
			//con ayuda de estos if nosotros sabremos en que posición se encuentra nuestro angulo
			//y dependiendo de su valor, se le asignará un valor adyacente para realizar este método 
			if ((157 <= angulo.at<float>(i, j) <= 180) || (0 <= angulo.at<float>(i, j) < 22)) {
				r = sobel.at<uchar>(i, j - 1);
				x = sobel.at<uchar>(i, j + 1);
			}
			if (112 <= angulo.at<float>(i, j) < 157) {
				r = sobel.at<uchar>(i + 1, j + 1);
				x = sobel.at<uchar>(i - 1, j - 1);
			}
			if (67 <= angulo.at<float>(i, j) < 112) {
				r = sobel.at<uchar>(i - 1, j);
				x = sobel.at<uchar>(i + 1, j);
			}
			if (22 <= angulo.at<float>(i, j) < 67) {
				r = sobel.at<uchar>(i - 1, j + 1);
				x = sobel.at<uchar>(i+1, j - 1);
			}
			//por ultimo, si nuestro punto (i,j) no cumple con las siguientes condiciones, directamente 
			//se le asignará el valor de 0 para que se vea oscuro en la imagen
			if (sobel.at<uchar>(i, j) < r || sobel.at<uchar>(i, j) < x)
				imagenUMS.at<uchar>(i, j) = 0;
			else	
				imagenUMS.at<uchar>(i, j) = sobel.at<uchar>(i, j);
		}
	}
	//Devolvemos la imagen de Non-Maximum Suppression
	return imagenUMS;
}
//Por ultimo, una vez que ya tenemos la función de Non-Maximum Suppression, lo que debe mos hacer es 
//resaltar los valores de la imagen dependiendo de los umbrales que ingrese el usuario 
Mat funcionCanny(Mat imagenUMS, float umbralMenor, float umbralMayor) {
	//creamos la matriz en donde guardaremos el resultado de todo el proceso de Canny
	Mat ImagenNuevaUmbral(imagenUMS.rows, imagenUMS.cols, CV_8UC1, Scalar(0));
	//usamos 2 for para recorrer la imagen
	for (int i = 0; i < imagenUMS.rows; i++) {
		for (int j = 0; j < imagenUMS.cols; j++) {
			//si nuestro punto (i,j) es menor que el umbral menor ingresado por el usuario, lo mandamos a 0
			//ya que no cumple la condición
			if (imagenUMS.at<uchar>(i,j) < umbralMenor) {
				ImagenNuevaUmbral.at<uchar>(i, j) = 0;
			}
			//en caso contrario, si el valor (i,j) se encuentra entre el umbral mas bajo y el umbral máximo
			//le ingresamos a (i,j) el umbral máximo
			else if (imagenUMS.at<uchar>(i, j) >= umbralMenor && imagenUMS.at<uchar>(i, j) < umbralMayor) {
				ImagenNuevaUmbral.at<uchar>(i, j) = umbralMayor;
			}
			//para finalizar, si no cumple con ninguna de las otras, es decir, si es mayor que nuestro 
			//umbral máximo, se le ingresará el valor de 255 (blanco)
			else {
				ImagenNuevaUmbral.at<uchar>(i, j) = 255;
			}
		}
	}
	//y regresamos la imagen de todo el método de canny
	return ImagenNuevaUmbral;
}

int main() {
	//mandamos a llamar la función para guardar la imagen original 
	Mat imagen = obtenerImagen();
	//creamos la matriz en donde guardará la imagen en grises 
	Mat imagenEscalaDeGrises(imagen.rows, imagen.cols, CV_8UC1);

	//con ayuda de 2 for recorremos cada pixel de la imagen de los 3 canales y los multiplicamos por el valor
	//de la estandar para obener la imagen en grises
	for (int i = 0; i < imagen.rows; i++) {
		for (int j = 0; j < imagen.cols; j++) {
			imagenEscalaDeGrises.at<uchar>(i, j) = (((imagen.at<cv::Vec3b>(i, j)[0] * 0.114) + (imagen.at<cv::Vec3b>(i, j)[1] * 0.587) + (imagen.at<cv::Vec3b>(i, j)[2]) * 0.299));

		}
	}
	//inicializamos en 0 los valores que se le pediran al usuario para posteriormente guardarlos dependiendo 
	//de lo que ingrese
	int filasColumnas = 0;
	float sigma = 0;
	float umbralMenor = 0;
	float umbralMayor = 0;
	cout << "Ingrese el numero de filas y columnas: "; cin >> filasColumnas;
	cout << "Ingrese el sigma: "; cin >> sigma;
	cout << "Ingrese el umbral menor: "; cin >> umbralMenor;
	cout << "Ingrese el umbral mayor: "; cin >> umbralMayor;
	//creamos la matriz y mandamos a llamar la función para obtener el kernel para el filtro gaussiano
	Mat KernelGauss = gauss(filasColumnas, sigma);
	//estos 2 for solo nos sirven para imprimir en consola el valor de nuestro kernel para el filtro
	//gaussiano
	for (int i = 0; i < KernelGauss.rows; i++) { //filas
		for (int j = 0; j < KernelGauss.cols; j++) { //columnas
			cout << KernelGauss.at<float>(i, j) << " ";
		}
		cout << endl;
	}
	//creamos la matriz donde se guardará la imagen con los bordes y madnamos a llamar la función que hace esto
	Mat matrizConPadding = imagenBordes(imagenEscalaDeGrises, filasColumnas);
	//creamos todas las matrices en donde alojaremos las imagenes
	Mat ImagenConfiltro(imagen.rows, imagen.cols, CV_8UC1, Scalar(0));
	Mat sobel(imagen.rows, imagen.cols, CV_8UC1, Scalar(0));
	Mat sobelAngulo(ImagenConfiltro.rows, ImagenConfiltro.cols, CV_32FC1, Scalar(0));
	Mat imagenUMS(ImagenConfiltro.rows, ImagenConfiltro.cols, CV_8UC1, Scalar(0));
	Mat ImagenNuevaUmbral(imagenUMS.rows, imagenUMS.cols, CV_8UC1, Scalar(0));
	//mandamos a llamar las funciones para obtener el filtro gaussiano, sobel, angulo de sobel, y las dos imagenes
	//del método de canny
	filtroGaussiano(KernelGauss, matrizConPadding, filasColumnas, ImagenConfiltro);
	sobel = operadorSobel(ImagenConfiltro);
	sobelAngulo = angulo(ImagenConfiltro);
	imagenUMS = UMS(sobelAngulo, sobel);
	ImagenNuevaUmbral = funcionCanny(imagenUMS, umbralMenor, umbralMayor);
	//por ultimo mandamos a imprimir las imagenes con sus tamaños 
	imprimir(matrizConPadding, imagen, ImagenConfiltro, sobel, sobelAngulo,imagenUMS, ImagenNuevaUmbral);
	waitKey(0);
	return 1;
}

/*
	Examen del primer parcial

	Código hecho por Rodrigo Olarte Astudillo 
	Grupo 5BM1

*/