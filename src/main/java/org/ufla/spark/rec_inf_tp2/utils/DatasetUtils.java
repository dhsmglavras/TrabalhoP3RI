package org.ufla.spark.rec_inf_tp2.utils;

import org.apache.spark.sql.Dataset;
import org.apache.spark.sql.Row;
import org.apache.spark.sql.types.StructType;

/**
 * Responsável por realizar operações utéis para dataset.
 * 
 * @author carlos
 * @author douglas
 * @author italo
 */
public class DatasetUtils {

	/**
	 * Recupera o índice de uma determinada coluna de um dataset
	 * 
	 * @param dataset
	 *            dataset a ser recuperado índice de coluna
	 * @param coluna
	 *            nome da coluna a ser recuperado o índice
	 * @return índice da determinada coluna do dataset
	 */
	public static int getIndiceColuna(Dataset<Row> dataset, String coluna) {
		return getIndiceColuna(dataset.schema(), coluna);
	}

	/**
	 * Recupera o índice de uma determinada coluna de um esquema
	 * 
	 * @param esquema
	 *            esquema a ser recuperado índice de coluna
	 * @param coluna
	 *            nome da coluna a ser recuperado o índice
	 * @return índice da determinada coluna do esquema
	 */
	public static int getIndiceColuna(StructType esquema, String coluna) {
		String[] colunas = esquema.fieldNames();
		for (int i = 0; i < colunas.length; i++) {
			if (colunas[i].equals(coluna)) {
				return i;
			}
		}
		return colunas.length;
	}

	/**
	 * Mostra uma linha de um dataset na tela as colunas ficam entre tags com o nome
	 * destas colunas.
	 * 
	 * @param row
	 *            linha a ser mostrada
	 * @param esquema
	 *            esquema da linha a ser mostrada
	 */
	public static void printRow(Row row, StructType esquema) {
		String[] colunas = esquema.fieldNames();
		System.out.println("\n----PRINT ROW----");
		for (int i = 0; i < row.length(); i++) {
			System.out.println("<" + colunas[i] + ">");
			System.out.println(row.get(i));
			System.out.println("</" + colunas[i] + ">");
		}
		System.out.println("\n----END PRINT ROW----\n");
	}

}
