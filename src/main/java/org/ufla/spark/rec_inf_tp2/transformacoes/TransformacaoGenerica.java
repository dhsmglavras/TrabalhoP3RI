package org.ufla.spark.rec_inf_tp2.transformacoes;

import org.apache.spark.api.java.function.MapFunction;
import org.apache.spark.ml.Transformer;
import org.apache.spark.ml.param.ParamMap;
import org.apache.spark.sql.Dataset;
import org.apache.spark.sql.Encoders;
import org.apache.spark.sql.Row;
import org.apache.spark.sql.types.DataTypes;
import org.apache.spark.sql.types.StructField;
import org.apache.spark.sql.types.StructType;
import org.ufla.spark.rec_inf_tp2.utils.DatasetUtils;

/**
 * Responsável por aplicar uma transformação genérica em uma string de uma
 * determinada coluna de um dataset e salvar a string transformada uma
 * determinada coluna . Aplica transformação em todo dataset.
 * 
 * @author carlos
 * @author douglas
 * @author italo
 */
public abstract class TransformacaoGenerica<F extends MapFunction<Row, Row>> extends Transformer {

	private static final long serialVersionUID = 1L;
	/**
	 * Nome da coluna de entrada da transformação.
	 */
	private String colunaEntrada;
	/**
	 * Nome da coluna de saída da transformação.
	 */
	private String colunaSaida;
	/**
	 * Esquema do dataset quando entra trasformação.
	 */
	private StructType esquemaEntrada;
	/**
	 * Esquema do dataset após a trasformação.
	 */
	protected StructType novoSaida;
	/**
	 * String representando uma instância da transformação.
	 */
	private String uid;

	/**
	 * Um contador de transformações instânciadas.
	 */
	private static int n = 0;

	/**
	 * Recupera o nome da transformação.
	 * 
	 * @return nome da transformação
	 */
	protected abstract String getLabeluid();

	@Override
	public String uid() {
		return uid;
	}

	public TransformacaoGenerica() {
		uid = getLabeluid() + "_" + this.hashCode() + "_" + n;
		n++;
	}

	/**
	 * Cria uma instância da transformação.
	 * 
	 * @return uma instância da transformação
	 */
	public abstract TransformacaoGenerica<?> criarTransformacao();

	/**
	 * Cria uma instância da função que aplica a transformação nas linhas do
	 * dataset.
	 * 
	 * @param indiceColEntrada
	 *            índice da coluna de entrada da transformação
	 * @param indiceColSaida
	 *            índice da coluna de saída da transformação
	 * @return instância da função que aplica a transformação nas linhas do dataset
	 */
	public abstract F criarFuncao(int indiceColEntrada, int indiceColSaida);

	@Override
	public Transformer copy(ParamMap arg0) {
		TransformacaoGenerica<?> copia = criarTransformacao().setColunaEntrada(colunaEntrada)
				.setColunaSaida(colunaSaida).setEsquemaEntrada(esquemaEntrada);
		copia.novoSaida = novoSaida;
		return copia;
	}

	/**
	 * Verifica se os índices da coluna de entrada e saída são válidos para um
	 * esquema com uma determinada quantidade de colunas.
	 * 
	 * @param qtdColunas
	 *            quantidade de colunas do esquema
	 * @param indiceColEntrada
	 *            índice da coluna de entrada da transformação
	 * @param indiceColSaida
	 *            índice da coluna de saída da transformação
	 * @return true, se os índices da coluna de entrada e saída são válidos para o
	 *         esquema, caso contrário, false
	 */
	public boolean eValido(int qtdColunas, int indiceColEntrada, int indiceColSaida) {
		return indiceColEntrada >= 0 && indiceColEntrada < qtdColunas && indiceColSaida >= 0
				&& indiceColSaida <= qtdColunas;
	}

	/**
	 * Cria o esquema que o dataset terá após a aplicação da transformação.
	 */
	public void criarEsquemaSaida() {
		if (esquemaEntrada == null) {
			throw new RuntimeException("Esquema antigo não foi definido em TransformaçãoMinúscula!");
		}
		int indiceColEntrada = DatasetUtils.getIndiceColuna(esquemaEntrada, colunaEntrada);
		int indiceColSaida = DatasetUtils.getIndiceColuna(esquemaEntrada, colunaSaida);
		int qtdColunas = esquemaEntrada.length();
		
		if (!eValido(qtdColunas, indiceColEntrada, indiceColSaida)) {
			throw new RuntimeException("Coluna de entrada/saida incorreta em TransformaçãoMinúscula!");
		}
		int novaQtdColunas = qtdColunas;
		if (qtdColunas == indiceColSaida) {
			novaQtdColunas++;
		}
		StructField[] novosCampos = new StructField[novaQtdColunas];
		StructField[] antigosCampos = esquemaEntrada.fields();
		StructField estruturaSaida = DataTypes.createStructField(colunaSaida, DataTypes.StringType, true);
		for (int i = 0; i < qtdColunas; i++) {
			novosCampos[i] = antigosCampos[i];
		}
		novosCampos[indiceColSaida] = estruturaSaida;
		novoSaida = DataTypes.createStructType(novosCampos);
	}

	/**
	 * Aplica a transformação no dataset e retorna o dataset transformado.
	 */
	@SuppressWarnings("unchecked")
	@Override
	public Dataset<Row> transform(Dataset<?> dataset) {
		Dataset<Row> datasetRow = (Dataset<Row>) dataset;
		esquemaEntrada = datasetRow.schema();
		criarEsquemaSaida();
		int indiceColEntrada = DatasetUtils.getIndiceColuna(esquemaEntrada, colunaEntrada);
		int indiceColSaida = DatasetUtils.getIndiceColuna(esquemaEntrada, colunaSaida);
		return dataset.sparkSession().createDataFrame(
				datasetRow.map(criarFuncao(indiceColEntrada, indiceColSaida), Encoders.bean(Row.class)).rdd(),
				novoSaida);
	}

	/**
	 * Retorna o esquema após a transformação.
	 */
	@Override
	public StructType transformSchema(StructType arg0) {
		if (novoSaida == null) {
			criarEsquemaSaida();
		}
		return novoSaida;
	}

	/**
	 * Recupera o nome da coluna de entrada da transformação.
	 * 
	 * @return o nome da coluna de entrada da transformação.
	 */
	public String getColunaEntrada() {
		return colunaEntrada;
	}

	/**
	 * Define o nome da coluna de entrada da transformação.
	 * 
	 * @param colunaEntrada
	 *            nome da coluna de entrada da transformação.
	 * @return a própria transformação, após definir o nome da coluna de entrada
	 */
	public TransformacaoGenerica<?> setColunaEntrada(String colunaEntrada) {
		this.colunaEntrada = colunaEntrada;
		return this;
	}

	/**
	 * Recupera o nome da coluna de saída da transformação.
	 * 
	 * @return o nome da coluna de saída da transformação.
	 */
	public String getColunaSaida() {
		return colunaSaida;
	}

	/**
	 * Define o nome da coluna de saída da transformação.
	 * 
	 * @param colunaSaida
	 *            nome da coluna de saída da transformação.
	 * @return a própria transformação, após definir o nome da coluna de saída
	 */
	public TransformacaoGenerica<?> setColunaSaida(String colunaSaida) {
		this.colunaSaida = colunaSaida;
		return this;
	}

	/**
	 * Recupera o esquema de entrada do dataset que será aplicada a transformação.
	 * 
	 * @return esquema de entrada do dataset que será aplicada a transformação.
	 */
	public StructType getEsquemaEntrada() {
		return esquemaEntrada;
	}

	/**
	 * Define o esquema de entrada do dataset que será aplicada a transformação.
	 * 
	 * @param esquemaEntrada
	 *            esquema de entrada do dataset que será aplicada a transformação.
	 * @return a própria transformação, após definir o esquema de entrada do dataset
	 *         que será aplicada a transformação.
	 */
	public TransformacaoGenerica<?> setEsquemaEntrada(StructType esquemaEntrada) {
		this.esquemaEntrada = esquemaEntrada;
		return this;
	}
}