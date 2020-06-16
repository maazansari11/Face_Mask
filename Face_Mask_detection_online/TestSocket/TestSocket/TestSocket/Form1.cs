using System;
using System.Collections.Generic;
using System.ComponentModel;
using System.Data;
using System.Drawing;
using System.IO;
using System.Linq;
using System.Net;
using System.Net.Sockets;
using System.Text;
using System.Threading.Tasks;
using System.Windows.Forms;
using TestSocket;

namespace TestSocket
{
    public partial class Form1 : Form
    {
        Socket socket = null;
        public Form1()
        {
            InitializeComponent();
        }

        public  void StartClient()
        {
            // Data buffer for incoming data.  
            byte[] bytes = new byte[9048];

            // Connect to a remote device.  

            try
            {
                // Establish the remote endpoint for the socket.  
                // This example uses port 11000 on the local computer.  
                IPHostEntry ipHostInfo = Dns.GetHostEntry(Dns.GetHostName());
                IPAddress ipAddress = ipHostInfo.AddressList[2];
                
                IPEndPoint remoteEP = new IPEndPoint(ipAddress, 5005);

                // Create a TCP/IP  socket.  
                socket = new Socket(ipAddress.AddressFamily,
                    SocketType.Stream, ProtocolType.Tcp);

                // Connect the socket to the remote endpoint. Catch any errors.  
                try
                {
                    socket.Connect(remoteEP);

                    Console.WriteLine("Socket connected to {0}",
                        socket.RemoteEndPoint.ToString());

                    // Encode the data string into a byte array.  
                    // camera_url ~ gstreamer_flag ~ usecase_id
                    byte[] msg = Encoding.ASCII.GetBytes("0~0~0");

                    // Send the data through the socket.  
                    int bytesSent = socket.Send(msg);
                    string dataStr = "";
                    // Receive the response from the remote device.  
                    while (true)
                    {
                        int bytesRec = socket.Receive(bytes);
                        //Console.WriteLine("Echoed test = {0}",bytesRec);
                        dataStr += Encoding.ASCII.GetString(bytes, 0, bytesRec);
                        //Console.WriteLine(dataStr);
                        /*if (bytes[bytesRec - 4] == 'E' & bytes[bytesRec - 3] == 'N' & bytes[bytesRec - 2] == 'D' & bytes[bytesRec - 1] == '!')
                        {
                            Console.WriteLine("Echoed test = {0}", Encoding.ASCII.GetString(bytes, 0, bytesRec));
                            Console.WriteLine("end found " + bytesRec);
                            break;
                        }*/
                        if (dataStr.Contains("END!"))
                        {
                            int index2 = dataStr.IndexOf("END!");
                            //Console.WriteLine("End found at::::" + index2);
                            int totLen = dataStr.Length;
                            //Console.WriteLine("length::::" + dataStr.Length);
                            //string mainData = dataStr.Substring(0, index2);
                            //if (index2 == mainData.Length)
                            //{
                            //    dataStr = dataStr.Substring(0, index2);
                            //}
                            //else
                            //{
                            string mainData = dataStr.Substring(0, index2);
                            //Console.WriteLine("main data length::::" + mainData.Length);
                            Image image = byteArrayToImage(mainData);
                            //pictureBox1.Image = image;
                            if (pictureBox1.Image != null) { pictureBox1.Image.Dispose(); }
                            pictureBox1.Image = (Image)image.Clone();
                            pictureBox1.Update();
                            //image.Dispose();
                            if (dataStr.Length == index2 + 4)
                            {
                                dataStr = "";
                            }
                            else if (dataStr.Length > (index2 + 4))
                            {
                                //Console.WriteLine("TEST TEST");
                                //Console.WriteLine("TEST TEST" + (index2 + 4));
                                string dataStr1 = dataStr.Substring(index2 + 4, (totLen - (index2 + 4)));
                                dataStr = dataStr1;
                            }
                            else
                            {

                            }
                            //}
                            //Console.WriteLine("remaining string::" + dataStr);
                        }

                    }

                    // Release the socket.  
                    socket.Shutdown(SocketShutdown.Both);
                    socket.Close();

                }
                catch (ArgumentNullException ane)
                {
                    Console.WriteLine("ArgumentNullException : {0}", ane.ToString());

                }
                catch (SocketException se)
                {
                    Console.WriteLine("SocketException : {0}", se.ToString());
                }
                catch (Exception e)
                {
                    Console.WriteLine("Unexpected exception : {0}", e.ToString());
                }

            }
            catch (Exception e)
            {
                Console.WriteLine(e.ToString());
            }
        }
        public byte[] GetBytes(string str)
        {
            byte[] bytes = new byte[str.Length * sizeof(char)];
            System.Buffer.BlockCopy(str.ToCharArray(), 0, bytes, 0, bytes.Length);
            return bytes;
        }

        public  Image  byteArrayToImage(string strDataImage)
        {

            byte[] bytedata = GetBytes(strDataImage);
            return Base64StringToBitmap(strDataImage);
            //return Image.FromStream(new MemoryStream(Convert.FromBase64String(strDataImage)));
            //return ConvertByteArrayToImage(bytedata);
        }

        public static Bitmap Base64StringToBitmap(string base64String)
        {
            Bitmap bmpReturn = null;
            //Convert Base64 string to byte[]
            byte[] byteBuffer = Convert.FromBase64String(base64String);
            MemoryStream memoryStream = new MemoryStream(byteBuffer);

            memoryStream.Position = 0;

            bmpReturn = (Bitmap)Bitmap.FromStream(memoryStream);

            memoryStream.Close();
            memoryStream = null;
            byteBuffer = null;

            return bmpReturn;
        }

        public Image ConvertByteArrayToImage(byte[] byteArrayIn)
        {
            using (MemoryStream ms = new MemoryStream(byteArrayIn))
            {
                return Image.FromStream(ms);
            }
        }

        //Bitmap GetBitmap(byte[] buf)
        //{
        /*Int16 width = 3072;// BitConverter.ToInt16(buf, 18);
        Int16 height = 1728;// BitConverter.ToInt16(buf, 22);
        Console.WriteLine(width);
        Console.WriteLine(height);
        Image< Bgr, Byte > image = new Image<Bgr, Byte>(width, height); //specify the width and height here
        image.Bytes = buf; //your byte array
        new Mat(200, 400, DepthType.Cv8U, 3);
        Mat m = image.Mat;
        Bitmap m1 = m.ToImage<Bgr, byte>().Bitmap;
        //Mat m = new Mat();
        //m = buf.
        //capture.Retrieve(m);

        //bitmap = new Bitmap(m1, new Size(m1.Width / 4, m1.Height / 4));


        /*int imageSize = width * height * 4;
        int headerSize = BitConverter.ToInt16(buf, 10);

        System.Diagnostics.Debug.Assert(imageSize == buf.Length - headerSize);

        int offset = headerSize;
        for (int y = 0; y < height; y++)
        {
            for (int x = 0; x < width; x++)
            {
                bitmap.SetPixel(x, height - y - 1, Color.FromArgb(buf[offset + 3], buf[offset], buf[offset + 1], buf[offset + 2]));
                offset += 4;
            }
        }*/
        //return m1;
        //}

        private void btnStart_Click(object sender, EventArgs e)
        {
            try
            {
                StartClient();
            }
            catch(Exception ex)
            {
                Console.WriteLine(ex.ToString());
            }
        }

        private void btnStop_Click(object sender, EventArgs e)
        {
            // Release the socket.  
            if (socket != null)
            {
                socket.Shutdown(SocketShutdown.Both);
                socket.Close();
            }
        }
    }
}
